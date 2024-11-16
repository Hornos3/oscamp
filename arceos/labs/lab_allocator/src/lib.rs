#![no_std]

#![feature(core_intrinsics)]
#![feature(ptr_sub_ptr)]
extern crate alloc;

use core::alloc::Layout;
use core::mem::transmute;
use core::ptr::NonNull;
use allocator::{AllocError, AllocResult, BaseAllocator, ByteAllocator, PageAllocator};
use log::{debug, error, info};

#[allow(dead_code)]
const PAGE_SIZE: usize = 0x1000;
#[allow(dead_code)]
const MIN_GROUP_UNIT: usize = 0x200;
#[allow(dead_code)]
const EARLY_ALLOCATOR_MIN_SIZE: usize = 0x8000;
#[allow(dead_code)]
const SIZE_CLASSES_COUNT: usize = 20;

const SIZE_CLASSES: [u8; SIZE_CLASSES_COUNT] = [
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 12, 15, 21, 25, 31,
    42, 63, 84, 127, 254
];

/// group header size: 4 bytes
/// total group size of each size_class:
/// 1 (0x10): 0x200, 31 slots
/// 2 (0x20): 0x200, 15 slots
/// 3 (0x30): 0x200, 10 slots
/// 4 (0x40): 0x200, 7 slots
/// 5 (0x50): 0x200, 6 slots
/// 6 (0x60): 0x200, 5 slots
/// 7 (0x70): 0x400, 8 slots
/// 8 (0x80): 0x400, 7 slots
/// 9 (0x90): 0x400, 6 slots
/// 10 (0xa0): 0x400, 6 slots
/// 12 (0xc0): 0x400, 5 slots
/// 15 (0xf0): 0x400, 4 slots
/// 21 (0x150): 0x800, 6 slots
/// 25 (0x190): 0x800, 5 slots
/// 31 (0x1f0): 0x800, 4 slots
/// 42 (0x2a0): 0x800, 3 slots
/// 63 (0x3f0): 0x800, 2 slots
/// 84 (0x540): 0x1000, 3 slots
/// 127 (0x7f0): 0x1000, 2 slots
/// 254 (0xfe0): 0x1000, 1 slots
///
const SLOT_COUNT: [u8; SIZE_CLASSES_COUNT] = [
    30, 15, 10, 7, 6, 5, 8, 7,
    6, 6, 5, 4, 6, 5, 4, 3, 2, 3, 2, 1
];

const INIT_AVAIL_MASK: [u16; SIZE_CLASSES_COUNT] = [
    0x3FFF, 0x7F, 0x3FF, 0x7F, 0x3F, 0x1F, 0xFF, 0x7F,
    0x3F, 0x3F, 0x1F, 0xF, 0x3F, 0x1F, 0xF, 0x7, 0x3, 0x7, 0x3, 0x1
];

const GROUP_TOTAL_SIZE: [u16; SIZE_CLASSES_COUNT] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8
];


/// 内存分配系统设计原理：
/// 页分配和小块分配共同作用，小块分配参考musl libc分配原理，设计上更加轻量化，但分配数量较多时会导致较大的时间开销。
/// 整个一大块内存空间需要使用1/4096来保存分配状态bitmask，一个位用于记录0x200大小的空间的使用情况。
/// 无法分配超过0x1_000_000，即大小超过0x1000个页的连续空间。
///
/// 整个内存分布情况：
/// | 1 bitmask page | 0x1000 pages | 1 bitmask page | 1000 pages | ... 交替
pub struct LabByteAllocator {
    start: usize,
    size: usize,
    bitmap_page_num: usize,
    alloc_top: usize,

    // byte allocator elements, 19*2 pointers for start and end pointer of 19 linked lists
    available_groups: [usize; SIZE_CLASSES_COUNT],     // pointer
}

#[derive(Copy, Clone)]
struct EAGroup {
    size_class: u8,
    is_head: bool,
    available_mask: u16,    // bitmap
    prev_group: usize,      // prev pointer
    next_group: usize,      // next pointer
}

impl EAGroup {
    pub fn new(size_class: u8) -> Self {
        assert!(size_class < SIZE_CLASSES_COUNT as u8);
        Self {
            size_class,
            available_mask: INIT_AVAIL_MASK[size_class as usize],
            is_head: false,
            next_group: 0,
            prev_group: 0
        }
    }

    pub fn end_addr(&self) -> NonNull<u8> {
        unsafe {
            NonNull::new((transmute::<*const EAGroup, usize>(self)
                + GROUP_TOTAL_SIZE[self.size_class as usize] as usize * MIN_GROUP_UNIT) as *mut u8).unwrap()
        }
    }

    pub fn get_chunk_addr(&self, idx: usize) -> Option<NonNull<u8>> {
        assert!(idx < SLOT_COUNT[self.size_class as usize].into());
        unsafe {
            let self_head: *mut u8 = transmute(self);
            let chunk_head: *mut u8 = self_head.add(0x20) as *mut u8;
            NonNull::new(chunk_head.add(SIZE_CLASSES[self.size_class as usize] as usize * 0x10 * idx))
        }
    }

    pub fn alloc_one(&mut self) -> Option<NonNull<u8>> {
        let avail_idx = first_avail(self.available_mask);
        if avail_idx == 16 { None } else {
            self.available_mask ^= 1 << avail_idx;
            // if it's full, it needs to be chained out
            if self.available_mask == 0 {
                self.unlink();
            }

            self.get_chunk_addr(avail_idx)
        }
    }

    // if the return value is true, this group was unlinked and allocator may need to change the top
    pub fn free_one(&mut self, idx: usize) -> bool {
        assert!(idx < SLOT_COUNT[self.size_class as usize] as usize);
        assert_eq!((1 << idx) as u16 & self.available_mask, 0, "Error, double free detected");

        self.available_mask ^= 1 << idx;

        if self.available_mask == INIT_AVAIL_MASK[self.size_class as usize] {
            // There is only one possibility: there is only 1 slot in this group
            if self.prev_group == 0 {
                assert_eq!(SLOT_COUNT[self.size_class as usize], 1);
                return true;
            }
            self.unlink();
            true
        } else { false }
    }

    pub fn init_offsets(&self) {
        unsafe {
            let mut offset_ptr: *mut u16 = transmute(self);
            offset_ptr = offset_ptr.add(0xF);
            let mut offset = 0;
            for _ in 0..SLOT_COUNT[self.size_class as usize] {
                offset_ptr.write(offset);
                // for flushing cache
                if offset_ptr as usize == 0xffffffc08028080e {
                    error!("{:p}!: {:#x}", offset_ptr, offset_ptr.read());
                }
                offset_ptr = offset_ptr.add(SIZE_CLASSES[self.size_class as usize] as usize * 0x10 / 2);
                offset += SIZE_CLASSES[self.size_class as usize] as u16;
            }
        }
    }

    pub fn unlink(&mut self) {
        assert_ne!(self.prev_group, 0);
        let next: *mut EAGroup = self.next_group as *mut EAGroup;
        if self.next_group != 0 {
            unsafe { next.as_mut().unwrap().prev_group = self.prev_group; }
        }

        if self.is_head {
            let allocator: *mut LabByteAllocator = self.prev_group as *mut LabByteAllocator;
            unsafe {
                assert_eq!(allocator.read().available_groups[self.size_class as usize],
                           transmute::<&mut EAGroup, usize>(self).clone());
                allocator.as_mut().unwrap().available_groups[self.size_class as usize] = next as usize;
                if self.next_group != 0 {
                    next.as_mut().unwrap().is_head = true;
                }
            }
            self.is_head = false;
        } else {
            let prev: *mut EAGroup = self.prev_group as *mut EAGroup;
            unsafe {
                prev.as_mut().unwrap().next_group = next as usize;
                if self.next_group != 0 {
                    next.as_mut().unwrap().prev_group = prev as usize;
                }
            }
        }

        self.next_group = 0;
        self.prev_group = 0;
    }
}

impl LabByteAllocator {
    pub const fn new() -> Self {
        Self {
            start: 0,
            size: 0,
            bitmap_page_num: 0,
            alloc_top: 0,
            available_groups: [0; SIZE_CLASSES_COUNT]
        }
    }

    pub fn get_bitmask_start(&self) -> *mut u8 {
        self.start as *mut u8
    }

    pub fn get_bitmask_end(&self) -> *mut u8 {
        (self.start
            + (PAGE_SIZE + 1) * PAGE_SIZE * (self.bitmap_page_num - 1)      // previous pages and bitmap pages
            + ((self.size / PAGE_SIZE) % (PAGE_SIZE + 1) - 1)) as *mut u8   // remaining pages
    }

    pub fn get_bitmask_current_top(&self) -> (*mut u8, u8) {
        self.addr_to_bitmask(self.alloc_top as *mut u8)
    }

    pub fn get_bitmask_current_page_top(&self) -> *mut u8 {
        let top = self.get_bitmask_current_top();
        if top.1 != 0 { unsafe { top.0.add(1) } }
        else { top.0 }
    }

    pub fn get_bitmask_next(&self, bitmask: *mut u8) -> Option<*mut u8> {
        if bitmask.gt(&(self.get_bitmask_end())) { return None }

        if bitmask as usize % PAGE_SIZE == PAGE_SIZE - 1 {
            self.next_bitmask_page_start(bitmask)
        } else {
            unsafe { Some(bitmask.add(1)) }
        }
    }

    pub fn get_bitmask_prev(&self, bitmask: *mut u8) -> Option<*mut u8> {
        if bitmask.eq(&self.get_bitmask_start()) { return None }

        if bitmask as usize % PAGE_SIZE == 0 {
            self.prev_bitmask_page_end(bitmask)
        } else {
            unsafe { Some(bitmask.sub(1)) }
        }
    }

    pub fn alloc_unused_group_space(&mut self, size_class: u8) -> Option<*mut u8> {
        let mut ptr = self.get_bitmask_start();

        let mut ret_val = None;

        'outer: while ptr.le(&self.get_bitmask_end()) {
            unsafe {
                match size_class {
                    _ if GROUP_TOTAL_SIZE[size_class as usize] == 8 => {
                        if ptr as usize > self.addr_to_bitmask(self.alloc_top as *mut u8).0 as usize {
                            debug!("{:p} ({:p}), {:#x}", ptr,
                                self.bitmask_to_addr(ptr, 0),
                                ptr.read());
                        }
                        if ptr.read() == 0xff {
                            ptr.write(0);
                            ret_val = Some(self.bitmask_to_addr(ptr, 0));
                            break 'outer;
                        }
                    },
                    _ if GROUP_TOTAL_SIZE[size_class as usize] == 4 => {
                        if ptr.read() & 0xf == 0xf {
                            ptr.write(ptr.read() ^ 0xf);
                            ret_val = Some(self.bitmask_to_addr(ptr, 0));
                            break 'outer;
                        }
                        else if ptr.read() == 0xf0 {
                            ptr.write(0);
                            ret_val = Some(self.bitmask_to_addr(ptr, 4));
                            break 'outer;
                        }
                    },
                    _ if GROUP_TOTAL_SIZE[size_class as usize] == 2 => {
                        let mut mask = 3;
                        let value = ptr.read();
                        for i in 0..4 {
                            if value & mask == mask {
                                ptr.write(ptr.read() ^ mask);
                                ret_val = Some(self.bitmask_to_addr(ptr, i*2));
                                break 'outer;
                            }
                            mask <<= 2;
                        }
                    },
                    _ if GROUP_TOTAL_SIZE[size_class as usize] == 1 => {
                        let mut mask = 1;
                        let value = ptr.read();
                        for i in 0..8 {
                            if value & mask == mask {
                                ptr.write(ptr.read() ^ mask);
                                ret_val = Some(self.bitmask_to_addr(ptr, i));
                                break 'outer;
                            }
                            mask <<= 1;
                        }
                    },
                    _ => {
                        panic!()
                    }
                }

                ptr = self.get_bitmask_next(ptr)?;
            }
        }

        unsafe {
            match ret_val {
                Some(group) => {
                    if group as usize >= self.alloc_top {
                        self.alloc_top = group as usize +
                            MIN_GROUP_UNIT * GROUP_TOTAL_SIZE[size_class as usize] as usize;
                    }
                    let group_object = (group as *mut EAGroup).as_mut().unwrap();
                    group_object.size_class = size_class;
                    group_object.prev_group = 0;
                    group_object.next_group = 0;
                    group_object.is_head = false;
                    group_object.available_mask = INIT_AVAIL_MASK[size_class as usize];
                    group_object.init_offsets();
                    Some(group)
                }
                None => {
                    debug!("Candidate address not found!!!");
                    None
                }
            }
        }
    }

    pub fn free_group_space(&mut self, group: *mut EAGroup) {
        // todo: check the validity of group address

        let (bitmask_addr, offset) = self.addr_to_bitmask(group as *mut u8);
        unsafe {
            let mut mask = ((1 << GROUP_TOTAL_SIZE[group.read().size_class as usize]) - 1) as u8;
            assert_eq!(((mask as usize) << offset as usize) >> 8, 0);
            mask <<= offset;
            assert_eq!(bitmask_addr.read() & mask, 0);
            bitmask_addr.write(bitmask_addr.read() ^ mask);
        }
    }

    pub fn shrink_alloc_top(&mut self) {
        let (mut bitmask_addr, _) =
            self.addr_to_bitmask(self.alloc_top as *mut u8);
        unsafe {
            while bitmask_addr.read() == 0xff {
                bitmask_addr = self.get_bitmask_prev(bitmask_addr).unwrap();
            }
            let mut bit = 1;
            let mut inner_offset = 0;
            assert_ne!(bitmask_addr.read(), 0xff);
            while bitmask_addr.read() & bit == 0 && inner_offset < 8 {
                bit <<= 1;
                inner_offset += 1;
            }
            if inner_offset == 8 {
                bitmask_addr = self.get_bitmask_next(bitmask_addr).unwrap();
                inner_offset = 0;
            }
            self.alloc_top = self.bitmask_to_addr(bitmask_addr, inner_offset) as usize;
        }
    }

    pub fn alloc_unused_page_space(&mut self, size: usize) -> Option<NonNull<u8>> {
        assert!(self.get_bitmask_current_top().0 <= self.get_bitmask_end());
        let actual_size = align_up(size, PAGE_SIZE);

        let mut bitmask_ptr = self.get_bitmask_start() as usize;

        let mut found_page = 0;
        while bitmask_ptr < self.get_bitmask_current_page_top() as usize {
            unsafe {
                if (bitmask_ptr as *mut u8).read() == 0xff {
                    found_page += 1;
                } else {
                    found_page = 0;
                }
                if found_page == actual_size / PAGE_SIZE {
                    // space found
                    let mut fill_ptr = bitmask_ptr - found_page + 1;
                    for _ in 0..actual_size / PAGE_SIZE {
                        assert_eq!((fill_ptr as *mut u8).read(), 0xff);
                        (fill_ptr as *mut u8).write(0);
                        fill_ptr = fill_ptr + 1;
                    }

                    return Some(NonNull::new(self.bitmask_to_addr(
                        (bitmask_ptr - found_page + 1) as *mut u8, 0)))?
                }
            }

            bitmask_ptr = self.get_bitmask_next(bitmask_ptr as *mut u8).unwrap() as usize;
            if bitmask_ptr % PAGE_SIZE == 0 { found_page = 0; }
        }

        // not found below the top, we can only expand the top
        // we need to switch to a new bitmask page for continuous space
        if bitmask_ptr % PAGE_SIZE + actual_size / PAGE_SIZE > PAGE_SIZE {
            bitmask_ptr = self.next_bitmask_page_start(bitmask_ptr as *mut u8)? as usize;
        }

        let alloc_space_start = self.bitmask_to_addr(bitmask_ptr as *mut u8, 0);

        // recheck if the space is enough
        let mut check_ptr = bitmask_ptr as *mut u8;
        if (self.get_bitmask_end() as usize - check_ptr as usize) < actual_size / PAGE_SIZE {
            return None;
        }
        if (self.get_bitmask_end() as usize) < check_ptr as usize {
            while (check_ptr as usize) > self.get_bitmask_end() as usize {
                unsafe {
                    check_ptr = check_ptr.sub(1);
                }
            }
        }

        let mut fill_ptr = bitmask_ptr as *mut u8;
        for _ in 0..actual_size / PAGE_SIZE {
            unsafe {
                assert_eq!(fill_ptr.read(), 0xff);
                fill_ptr.write(0);
                fill_ptr = self.get_bitmask_next(fill_ptr).unwrap();
            }
        }
        self.alloc_top = alloc_space_start as usize + actual_size;
        Some(NonNull::new(alloc_space_start)?)
    }

    pub fn bitmask_to_addr(&self, bitmask_addr: *mut u8, inner_idx: u8) -> *mut u8 {
        assert!(inner_idx < 8);
        let bitmask_page_addr = bitmask_addr as usize / PAGE_SIZE * PAGE_SIZE;
        let inner_offset = (bitmask_addr as usize - bitmask_page_addr) * PAGE_SIZE +
            inner_idx as usize * MIN_GROUP_UNIT;
        let bitmask_page_idx = self.bitmask_idx(bitmask_addr) / PAGE_SIZE;

        (self.start + bitmask_page_idx * ((PAGE_SIZE + 1) * PAGE_SIZE) + PAGE_SIZE + inner_offset)
            as *mut u8
    }

    pub fn addr_to_bitmask(&self, addr: *mut u8) -> (*mut u8, u8) {
        let bitmask_page_idx = (addr as usize - self.start) / ((PAGE_SIZE + 1) * PAGE_SIZE);
        let bitmask_page_addr = self.start + bitmask_page_idx * ((PAGE_SIZE + 1) * PAGE_SIZE);
        let inner_offset = (addr as usize - self.start) % ((PAGE_SIZE + 1) * PAGE_SIZE) - PAGE_SIZE;
        let bitmask_byte_offset = inner_offset / PAGE_SIZE;
        let bitmask_bit_offset = (inner_offset % PAGE_SIZE) / MIN_GROUP_UNIT;

        ((bitmask_page_addr + bitmask_byte_offset) as *mut u8, bitmask_bit_offset as u8)
    }

    pub fn next_bitmask_page_start(&self, bitmask: *mut u8) -> Option<*mut u8> {
        if self.bitmask_idx(bitmask) / PAGE_SIZE == self.bitmap_page_num - 1 { None }
        else {
            Some(
                (bitmask as usize / PAGE_SIZE * PAGE_SIZE + (PAGE_SIZE * (PAGE_SIZE + 1)))
                    as *mut u8
            )
        }
    }

    pub fn prev_bitmask_page_end(&self, bitmask: *mut u8) -> Option<*mut u8> {
        if self.bitmask_idx(bitmask) / PAGE_SIZE == 0 { None }
        else {
            Some(
                (bitmask as usize / PAGE_SIZE * PAGE_SIZE - (PAGE_SIZE * PAGE_SIZE) - 1)
                    as *mut u8
            )
        }
    }

    pub fn bitmask_idx(&self, bitmask: *mut u8) -> usize {
        let bitmask_page = bitmask as usize / PAGE_SIZE * PAGE_SIZE;
        assert_eq!((bitmask_page - self.start) % ((PAGE_SIZE + 1) * PAGE_SIZE), 0);

        let bitmask_page_idx = (bitmask_page - self.start) / ((PAGE_SIZE + 1) * PAGE_SIZE);
        let bitmask_offset = bitmask as usize % PAGE_SIZE;

        bitmask_page_idx * PAGE_SIZE + bitmask_offset
    }

    pub fn chain_in(&mut self, group: *mut EAGroup) {
        unsafe {
            let idx = group.read().size_class;
            // the head group saves the EarlyAllocator pointer to prev_group
            group.as_mut().unwrap().prev_group = transmute::<&mut LabByteAllocator, usize>(self);

            if self.available_groups[idx as usize] == 0 {
                self.available_groups[idx as usize] = group as usize;
                group.as_mut().unwrap().is_head = true;

                return;
            } else {
                let head = (self.available_groups[idx as usize] as *mut EAGroup)
                    .as_mut().unwrap();
                head.is_head = false;
                head.prev_group = group as usize;
            }
            group.as_mut().unwrap().next_group = self.available_groups[idx as usize];
            group.as_mut().unwrap().is_head = true;

            self.available_groups[idx as usize] = group as usize;
        }
    }
}

impl BaseAllocator for LabByteAllocator {
    fn init(&mut self, start: usize, size: usize) {
        debug!("Start initializing global allocator, size = {:#x}", size);
        self.bitmap_page_num = align_up(size / PAGE_SIZE, PAGE_SIZE) / PAGE_SIZE;
        self.start = start;
        self.size = size;
        self.alloc_top = self.start + PAGE_SIZE;           // ptr as usize

        // clear bitmap, 1 for avail, 0 for using
        debug!("Initializing allocator bitmap...");
        let bitmap_end: *mut u8 = self.get_bitmask_end();
        let mut ptr: *mut u8 = self.start as *mut u8;
        unsafe {
            while ptr.lt(&bitmap_end) {
                ptr.write(0xFF);
                ptr = ptr.add(1);
            }
        }
    }

    fn add_memory(&mut self, start: usize, size: usize) -> AllocResult {
        // only continuous space add supported now
        assert_eq!(start, self.start + self.size);

        let mut bitmask = self.get_bitmask_end();
        self.size += size;
        let new_page_num = self.size / (PAGE_SIZE * (PAGE_SIZE + 1)) + 1;
        let added_page_num = new_page_num - self.bitmap_page_num;
        self.bitmap_page_num = new_page_num;
        for _ in 0..size / PAGE_SIZE - added_page_num {
            unsafe {
                bitmask.write(0xff);
                bitmask = self.get_bitmask_next(bitmask).unwrap();
            }
        }

        Ok(())
    }
}

impl ByteAllocator for LabByteAllocator {
    fn alloc(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        debug!("alloc {:p}, {:p}", self.get_bitmask_current_page_top(), self.get_bitmask_end());
        assert!(self.get_bitmask_current_page_top() <= self.get_bitmask_end());
        // let mut watch = 0xffffffc08028080e as *mut u16;
        // unsafe {
        //     debug!("{:p}: {:#x}", watch, watch.read());
        //
        //     let bitmask = self.addr_to_bitmask(watch as *mut u8);
        //     let mut bitmask_start = self.get_bitmask_start() as *mut usize;
        //     debug!("bitmask: {:p}, {:#x}", bitmask.0, bitmask.0.read());
        //     for _ in 0..10 {
        //         debug!("{:p}: {:#x}", bitmask_start, bitmask_start.read());
        //         bitmask_start = bitmask_start.add(1);
        //     }
        // }

        // if larger than the whole place
        if layout.size() > PAGE_SIZE * PAGE_SIZE {
            return Err(AllocError::InvalidParam);
        }

        // if larger than 0x1000,
        if layout.size() > ((SIZE_CLASSES[SIZE_CLASSES_COUNT - 1] as usize) << 4) - 0x20 {
            unsafe {
                return self.alloc_pages(
                    align_up(layout.size(), PAGE_SIZE) / PAGE_SIZE, PAGE_SIZE)
                    .map(|start|
                        NonNull::new(transmute::<usize, *mut u8>(start)).unwrap());
            }
        }

        // find avail group first
        let size_class = size_to_sc(layout.size());
        assert_ne!(size_class, 255);
        if self.available_groups[size_class as usize] != 0 {
            unsafe {
                let ag: *mut EAGroup = transmute(self.available_groups[size_class as usize]);
                let ret = ag.as_mut().unwrap().alloc_one().unwrap();
                info!("ALLOC SUCCESS: addr = \x1b[1;31m{:#x}\x1b[32m, size = {:#x}", ret.as_ptr() as usize, layout.size());
                return Ok(ret);
            }
        }

        // no avail group found, create a group
        let new_group = self.alloc_unused_group_space(size_class);
        if new_group.is_none() {
            info!("No memory available");
            return Err(AllocError::NoMemory)
        }

        // update the byte allocator bound
        unsafe {
            let new_group = new_group.unwrap() as *mut EAGroup;
            self.chain_in(new_group);

            let result = new_group.as_mut().unwrap().alloc_one().unwrap();
            info!("ALLOC SUCCESS: addr = \x1b[1;31m{:#x}\x1b[32m, size = {:#x}", result.as_ptr() as usize, layout.size());
            Ok(result)
        }
    }

    fn dealloc(&mut self, pos: NonNull<u8>, layout: Layout) {
        assert!(self.get_bitmask_current_page_top() <= self.get_bitmask_end());
        // let mut watch = 0xffffffc08028080e as *mut u16;
        // unsafe {
        //     debug!("{:p}: {:#x}", watch, watch.read());
        //
        //     let bitmask = self.addr_to_bitmask(watch as *mut u8);
        //     let mut bitmask_start = self.get_bitmask_start() as *mut usize;
        //     debug!("bitmask: {:p}, {:#x}", bitmask.0, bitmask.0.read());
        //     for _ in 0..10 {
        //         debug!("{:p}: {:#x}", bitmask_start, bitmask_start.read());
        //         bitmask_start = bitmask_start.add(1);
        //     }
        // }

        // check if it's in page allocator, if true, switch to free_page
        let addr = pos.as_ptr() as usize;
        if layout.size() > (SIZE_CLASSES[SIZE_CLASSES_COUNT - 1] as usize) << 4 {
            return self.dealloc_pages(addr, align_up(layout.size(), PAGE_SIZE) / PAGE_SIZE);
        }

        info!("Deallocating \x1b[1;33m{:#x}\x1b[1;32m, size = {:#x}", pos.as_ptr() as usize, layout.size());

        // find the group first
        let (group, idx) = find_group_of_chunk(pos);

        // if the group is previously full, this is the first free, chain it in
        unsafe {
            if group.read().available_mask == INIT_AVAIL_MASK[group.read().size_class as usize] {
                self.chain_in(group.as_ptr());
            }
        }

        // free it, unlink is done inside
        unsafe {
            let unlinked = group.as_ptr().as_mut().unwrap().free_one(idx);
            if unlinked {
                // release the bitmap
                self.free_group_space(group.as_ptr());
                // we may shrink the byte allocator top
                self.shrink_alloc_top();
            }
        }
    }

    fn total_bytes(&self) -> usize {
        // self.total_pages() * PAGE_SIZE
        PAGE_SIZE * 8
    }

    fn used_bytes(&self) -> usize {
        let used_pages = self.used_pages();
        let (top_byte, top_bit) = self.get_bitmask_current_top();

        if top_bit == 0 { used_pages * PAGE_SIZE }
        else { (used_pages - 1) * PAGE_SIZE + top_bit as usize * MIN_GROUP_UNIT }
    }

    fn available_bytes(&self) -> usize {
        self.total_bytes() - self.used_bytes()
    }
}

impl PageAllocator for LabByteAllocator {
    const PAGE_SIZE: usize = PAGE_SIZE;

    fn alloc_pages(&mut self, num_pages: usize, align_pow2: usize) -> AllocResult<usize> {
        assert_eq!(align_pow2, PAGE_SIZE);
        if num_pages > 0x1000 {
            info!("Cannot allocate a continuous space larger than 0x1_000_000");
            return Err(AllocError::InvalidParam);
        }

        match self.alloc_unused_page_space(num_pages * PAGE_SIZE) {
            Some(res) => {
                info!("Alloced page from \x1b[1;31m{:p}\x1b[1;32m, page num {:#x}", res.as_ptr(), num_pages);
                assert!((res.as_ptr() as usize) < self.alloc_top);
                Ok(res.as_ptr() as usize)
            },
            None => {
                Err(AllocError::NoMemory)
            }
        }
    }

    fn dealloc_pages(&mut self, pos: usize, num_pages: usize) {
        info!("Deallocating \x1b[1;31m{pos:#x}\x1b[1;32m, page_num = {num_pages:#x}");
        assert_eq!(pos % PAGE_SIZE, 0);
        let (mut bitmask, _) = self.addr_to_bitmask(pos as *mut u8);
        for _ in 0..num_pages {
            unsafe {
                assert_eq!(bitmask.read(), 0);
                assert!(bitmask.lt(&self.get_bitmask_end()));
                bitmask.write(0xff);
                bitmask = self.get_bitmask_next(bitmask).unwrap();
            }
        }
        self.shrink_alloc_top();
    }

    fn total_pages(&self) -> usize {
        self.size / PAGE_SIZE - self.bitmap_page_num
    }

    fn used_pages(&self) -> usize {
        let (top_byte, top_bit) = self.get_bitmask_current_top();

        if top_bit == 0 { self.bitmask_idx(top_byte) }
        else { self.bitmask_idx(top_byte) + 1 }
    }

    fn available_pages(&self) -> usize {
        self.total_pages() - self.used_pages()
    }
}

#[inline]
#[allow(dead_code)]
const fn align_down(pos: usize, align: usize) -> usize {
    pos & !(align - 1)
}

#[inline]
#[allow(dead_code)]
const fn align_up(pos: usize, align: usize) -> usize {
    (pos + align - 1) & !(align - 1)
}

fn size_to_sc(request: usize) -> u8 {
    let chunk_size = align_up(request + 2, 0x10) >> 4;
    for i in 0..SIZE_CLASSES_COUNT {
        if SIZE_CLASSES[i] as usize >= chunk_size { return i as u8; }
    }
    255
}

fn first_avail(group_bitmap: u16) -> usize {
    let mut b: u16 = 1;
    let mut ret = 0;
    while b & group_bitmap == 0 && ret < 16 { b <<= 1; ret += 1; }
    ret
}

// the second return value is the index of this chunk in this group
#[inline]
fn find_group_of_chunk(chunk_start: NonNull<u8>) -> (NonNull<EAGroup>, usize) {
    unsafe {
        let offset = (chunk_start.as_ptr() as *mut u16).sub(1);
        let offset_value = offset.read();
        let group = NonNull::new((offset as *mut u8).sub(((offset_value as usize) << 4) + 0x1e) as *mut EAGroup).unwrap();
        assert_eq!(group.as_ptr() as usize % 0x10, 0);
        let size_class = group.read().size_class;

        // some integrity check for group struct
        assert!(size_class < SIZE_CLASSES_COUNT as u8);
        assert_eq!(!INIT_AVAIL_MASK[group.read().size_class as usize] & group.read().available_mask, 0);

        let chunk_size = SIZE_CLASSES[size_class as usize] as u16;     // 0x10 as unit
        assert_eq!(offset_value % chunk_size, 0);

        (group, (offset_value / chunk_size) as usize)
    }
}