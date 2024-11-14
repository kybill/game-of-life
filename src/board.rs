use std::{collections::HashMap, fs::File, io::{BufReader, Read, Write}, time::Instant};
use cust::{function::Function, stream::Stream, prelude::*};

pub struct Board {
    chunks: HashMap<(i32, i32), Chunk>,
    pub true_positions: Vec<(i32, i32)>,
}

impl Board {
    pub fn new() -> Board {
        Board {
            chunks: HashMap::new(),
            true_positions: vec![],
        }
    }

    pub fn load(&mut self, file: &str) {
        let file_o = File::open(file).unwrap();
        let mut buf_reader = BufReader::new(file_o);
        let mut len_buf = [0u8; 8];
        buf_reader.read_exact(&mut len_buf).unwrap();
        let len = usize::from_be_bytes(len_buf);
        self.true_positions = vec![];
        
        for _u in 0..len {
            let mut cx_be = [0u8; 4];
            let mut cy_be = [0u8; 4];
            let mut data_be = [0u8; 8];
            let mut lx_be = [0u8; 1];
            let mut ux_be = [0u8; 1];
            let mut ly_be = [0u8; 1];
            let mut uy_be = [0u8; 1];

            buf_reader.read_exact(&mut cx_be).unwrap();
            buf_reader.read_exact(&mut cy_be).unwrap();
            buf_reader.read_exact(&mut data_be).unwrap();
            buf_reader.read_exact(&mut lx_be).unwrap();
            buf_reader.read_exact(&mut ux_be).unwrap();
            buf_reader.read_exact(&mut ly_be).unwrap();
            buf_reader.read_exact(&mut uy_be).unwrap();

            let cx = i32::from_be_bytes(cx_be);
            let cy = i32::from_be_bytes(cy_be);
            let data_u = u64::from_be_bytes(data_be);
            let lx = u8::from_be_bytes(lx_be); 
            let ux = u8::from_be_bytes(ux_be);
            let ly = u8::from_be_bytes(ly_be);
            let uy = u8::from_be_bytes(uy_be);

            let mut chunk = Chunk::new();
            chunk.data = data_u;
            chunk.lowest_x = lx;
            chunk.highest_x = ux;
            chunk.lowest_y = ly;
            chunk.highest_y = uy;
            
            self.chunks.insert((cx, cy), chunk);
        }

        self.true_positions = self.compress_chunk_array();
    }

    pub fn save(&mut self, file: &str) {
        self.chunks = self.decompress_chunk_array(&self.true_positions);

        // Format: No separators, just cx-cy-data for all chunks so i32-i32-u64
        let mut file_o = File::create(file).unwrap();
        let mut len_bytes = self.chunks.len().to_be_bytes();
        file_o.write_all(&mut len_bytes).unwrap();

        for (key, value) in &self.chunks {
            let mut data = vec![];
            data.extend_from_slice(&key.0.to_be_bytes());
            data.extend_from_slice(&key.1.to_be_bytes());
            data.extend_from_slice(&value.data.to_be_bytes());
            data.extend_from_slice(&value.lowest_x.to_be_bytes());
            data.extend_from_slice(&value.highest_x.to_be_bytes());
            data.extend_from_slice(&value.lowest_y.to_be_bytes());
            data.extend_from_slice(&value.highest_y.to_be_bytes());
            file_o.write_all(&mut data).unwrap();
        }
    }

    pub fn clone(&self) -> Board {
        Board {
            chunks: self.chunks.clone(),
            true_positions: self.true_positions.clone(),
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn flip_cell_chunk(&mut self, x: i32, y: i32) {
        let (cx, cy, ix, iy) = self.get_chunk_coords(x, y);
        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.flip_cell(ix, iy);
        if chunk.data == 0 {
            self.chunks.remove(&(cx, cy));
        }
    }

    pub fn set_cell_chunk(&mut self, x: i32, y: i32, state: bool) {
        let (cx, cy, ix, iy) = self.get_chunk_coords(x, y);
        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.set_cell(ix, iy, state);
        if !state && chunk.data == 0 {
            self.chunks.remove(&(cx, cy));
        }
    }

    pub fn flip_cell(&mut self, x: i32, y: i32) {
        if self.true_positions.contains(&(x, y)) {
            let index = self.true_positions.iter().position(|val| *val == (x, y)).unwrap();
            self.true_positions.swap_remove(index);
        } else {
            self.true_positions.push((x, y));
        }
    }

    pub fn get_or_create_chunk(&mut self, chunk_x: i32, chunk_y: i32) -> &mut Chunk { 
        let chunk_map = &mut self.chunks;

        if !chunk_map.contains_key(&(chunk_x, chunk_y)) {
            let new_chunk = Chunk::new();
            chunk_map.insert((chunk_x, chunk_y), new_chunk);
        }

        return chunk_map.get_mut(&(chunk_x, chunk_y)).unwrap();
    }

    pub fn get_chunk(&mut self, chunk_x: i32, chunk_y: i32) -> Option<&mut Chunk> {
        self.chunks.get_mut(&(chunk_x, chunk_y))
    }

    pub fn get_chunk_coords(&self, x: i32, y: i32) -> (i32, i32, i32, i32) { // chunk_x, chunk_y, internal_x, internal_y
        let chunk_x = if x >= 0 { x / 8 } else { (x + 1) / 8 - 1 };
        let chunk_y = if y >= 0 { y / 8 } else { (y + 1) / 8 - 1 };

        let internal_x = if x >= 0 { x % 8 } else { -(x+1) % 8 };
        let internal_y = if y >= 0 { y % 8 } else { -(y+1) % 8 };

        //println!("{} - {}, {} - {} - {} - {}", x, y, chunk_x, chunk_y, internal_x, internal_y);
        //let (rx, ry) = self.get_global_coords(chunk_x, chunk_y, internal_x, internal_y);
        //println!("Original: {} - {}, Recalculated: {} - {}", x, y, rx, ry);

        (chunk_x, chunk_y, internal_x, internal_y)
    }

    pub fn get_global_coords(&self, cx: i32, cy: i32, ix: i32, iy: i32) -> (i32, i32) {
        let x = if cx >= 0 { cx * 8 + ix } else { (cx + 1) * 8 - 1 - ix };
        let y = if cy >= 0 { cy * 8 + iy } else { (cy + 1) * 8 - 1 - iy };

        (x, y)
    }
    pub fn step_simulation_cpu(&mut self) -> (Instant, Instant, Instant, Instant) {
        let start_time = Instant::now();
        let mut time_data = (start_time, start_time, start_time, start_time);

        //let next_board = self.clone();
        let current_true_positions = &self.true_positions;
        let mut to_check = vec![];
        for (x, y) in current_true_positions {
            for i in -1..=1 {
                for j in -1..=1 {
                    let val = (x + i, y + j);
                    if !to_check.contains(&val) {
                        to_check.push(val);
                    }
                }
            }
        }
        let mut new_board = vec![];

        time_data.1 = Instant::now();

        // TODO: Apply game logic inside of modified_array using to_check coordinates
        for (x, y) in to_check {
            let mut neighbor_count = 0;

            for i in -1..=1 {
                for j in -1..=1 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    
                    if current_true_positions.contains(&(x + i, y + j)) {
                        neighbor_count += 1;
                    }
                }
            }

            /*if neighbor_count != 0 {
                println!("{}", neighbor_count);
            }*/

            if current_true_positions.contains(&(x, y)) {
                match neighbor_count {
                    ..=1 => {},
                    2..=3 => {
                        new_board.push((x, y));
                    },
                    4.. => {}
                }
            } else {
                if neighbor_count == 3 {
                    new_board.push((x, y));
                }
            }
        }

        time_data.2 = Instant::now();

        self.true_positions = new_board;

        time_data.3 = Instant::now();        
        return time_data;
    }

    pub fn step_simulation(&mut self, check_true: &Function<'_>, check_false: &Function<'_>, stream: &Stream) -> (Instant, Instant, Instant, Instant) {
        let start_time = Instant::now();
        let mut time_data = (start_time, start_time, start_time, start_time);

        if self.true_positions.len() == 0 {
            return time_data;
        }

        let (_, block_size_true) = check_true.suggested_launch_configuration(0, 0.into()).unwrap();
        let (_, block_size_false) = check_false.suggested_launch_configuration(0, 0.into()).unwrap();

        let current_true_positions = &self.true_positions;
        let mut true_cells_x = vec![0i32; current_true_positions.len()];
        let mut true_cells_y = vec![0i32; current_true_positions.len()];
        let mut false_cells_x = vec![];
        let mut false_cells_y = vec![];

        for i in 0..current_true_positions.len() {
            let (x, y) = current_true_positions[i];
            true_cells_x[i] = x;
            true_cells_y[i] = y;

            for j in -1..=1 {
                for k in -1..=1 {
                    if j == 0 && k == 0 {
                        continue;
                    }

                    let (xc, yc) = (x + j, y + k);
                    if !current_true_positions.contains(&(xc, yc)) {
                        false_cells_x.push(xc);
                        false_cells_y.push(yc);
                    }
                }
            }
        }

        let grid_size_true = (true_cells_x.len() as u32 + block_size_true - 1) / block_size_true;
        let grid_size_false = (false_cells_x.len() as u32 + block_size_false - 1) / block_size_false;

        time_data.1 = Instant::now();

        let tcx_gpu = true_cells_x.as_slice().as_dbuf().unwrap();
        let tcy_gpu = true_cells_y.as_slice().as_dbuf().unwrap();
        let fcx_gpu = false_cells_x.as_slice().as_dbuf().unwrap();
        let fcy_gpu = false_cells_y.as_slice().as_dbuf().unwrap();
        
        let mut out_true_x = vec![i32::MAX; true_cells_x.len()];
        let mut out_true_y = out_true_x.clone();
        let mut out_false_x = vec![i32::MAX; false_cells_x.len()];
        let mut out_false_y = out_false_x.clone();

        let true_out_gpu_x = out_true_x.as_slice().as_dbuf().unwrap();
        let true_out_gpu_y = out_true_y.as_slice().as_dbuf().unwrap();
        let false_out_gpu_x = out_false_x.as_slice().as_dbuf().unwrap();
        let false_out_gpu_y = out_false_y.as_slice().as_dbuf().unwrap();

        let mut nb_ct_true = vec![0i32; true_cells_x.len()];
        let mut nb_ct_false = vec![0i32; false_cells_x.len()];

        let tnbct_gpu = nb_ct_true.as_slice().as_dbuf().unwrap();
        let fnbct_gpu = nb_ct_false.as_slice().as_dbuf().unwrap();

        /*print!("True Cells: ");
        for i in 0..true_cells_x.len() {
            let (tx, ty) = (true_cells_x[i], true_cells_y[i]);
            print!("({}, {}), ", tx, ty);
        }
        print!("\nFalse Cells: ");
        for j in 0..false_cells_x.len() {
            let (fx, fy) = (false_cells_x[j], false_cells_y[j]);
            print!("({}, {}), ", fx, fy);
        }
        println!("");*/

        unsafe {
            launch!(
                check_true<<<grid_size_true, block_size_true, 0, stream>>>(
                    tcx_gpu.len(),
                    tcx_gpu.as_device_ptr(),
                    tcy_gpu.as_device_ptr(),
                    true_out_gpu_x.as_device_ptr(),
                    true_out_gpu_y.as_device_ptr(),
                    tnbct_gpu.as_device_ptr(),
                )
            ).unwrap();
            launch!(
                check_false<<<grid_size_false, block_size_false, 0, stream>>>(
                    fcx_gpu.len(),
                    fcx_gpu.as_device_ptr(),
                    fcy_gpu.as_device_ptr(),
                    tcx_gpu.len(),
                    tcx_gpu.as_device_ptr(),
                    tcy_gpu.as_device_ptr(),
                    false_out_gpu_x.as_device_ptr(),
                    false_out_gpu_y.as_device_ptr(),
                    fnbct_gpu.as_device_ptr(),
                )
            ).unwrap();
        }

        stream.synchronize().unwrap();

        true_out_gpu_x.copy_to(&mut out_true_x).unwrap();
        true_out_gpu_y.copy_to(&mut out_true_y).unwrap();
        false_out_gpu_x.copy_to(&mut out_false_x).unwrap();
        false_out_gpu_y.copy_to(&mut out_false_y).unwrap();
        tnbct_gpu.copy_to(&mut nb_ct_true).unwrap();
        fnbct_gpu.copy_to(&mut nb_ct_false).unwrap();

        time_data.2 = Instant::now();

        // Take gpu data and put it back into board
        let mut true_cells = vec![];
        for i in 0..out_true_x.len() {
            let (x, y) = (out_true_x[i], out_true_y[i]);
            if x < i32::MAX {
                true_cells.push((x, y));
            }
        }
        
        for j in 0..out_false_x.len() {
            let (x, y) = (out_false_x[j], out_false_y[j]);
            if x < i32::MAX && !true_cells.contains(&(x, y)) {
                true_cells.push((x, y));
            }
        }

        self.true_positions = true_cells;

        time_data.3 = Instant::now();
        return time_data;
    }

    pub fn decompress_chunk_array(&self, true_cells: &Vec<(i32, i32)>) -> HashMap<(i32, i32), Chunk> {
        let mut new_board = Board::new();

        for (ax, ay) in true_cells {
            new_board.set_cell_chunk(*ax, *ay, true);
        }

        return new_board.chunks;
    }

    pub fn compress_chunk_array(&self) -> Vec<(i32, i32)> {
        let mut true_coords = vec![];

        for (chunk_coords, chunk) in &self.chunks {
            for ix in 0..8 {
                for iy in 0..8 {
                    let (global_x, global_y) = self.get_global_coords(chunk_coords.0, chunk_coords.1, ix, iy);
                    
                    if chunk.get_cell(ix, iy) {
                        true_coords.push((global_x, global_y));
                    }
                }
            }
        }

        return true_coords;
    }

    fn get_bounds(&self) -> (i32, i32, i32, i32) {
        let mut keys = self.chunks.keys();
        let mut vals = self.chunks.values();

        let mut stop = false;

        let mut lower_x = 1000000;
        let mut upper_x = -1000000;
        let mut lower_y = 1000000;
        let mut upper_y = -1000000;

        while !stop {
            let key = keys.next();
            let val = vals.next();

            if key.is_none() || val.is_none() {
                stop = true;
                continue;
            }

            let (cx, cy) = *key.unwrap();
            let chunk = val.unwrap();

            let lowest_x = if cx >= 0 { chunk.lowest_x } else { chunk.highest_x } as i32;
            let highest_x = if cx >= 0 { chunk.highest_x } else { chunk.lowest_x } as i32;
            let lowest_y = if cy >= 0 { chunk.lowest_y } else { chunk.highest_y } as i32;
            let highest_y = if cy >= 0 { chunk.highest_y } else { chunk.lowest_y } as i32;

            let (lx, ly) = self.get_global_coords(cx, cy, lowest_x, lowest_y);
            let (ux, uy) = self.get_global_coords(cx, cy, highest_x, highest_y);

            if lx < lower_x {
                lower_x = lx;
            }
            if ux > upper_x {
                upper_x = ux;
            }
            if ly < lower_y {
                lower_y = ly;
            }
            if uy > upper_y {
                upper_y = uy;
            }
        }

        if lower_x == 1000000 || upper_x == -1000000 || lower_y == 1000000 || upper_y == -1000000 {
            return (0,0,0,0);
        }

        return (lower_x, upper_x, lower_y, upper_y);
    }
}

pub struct Chunk {
    data: u64,
    lowest_x: u8,
    highest_x: u8,
    lowest_y: u8,
    highest_y: u8
}

impl Clone for Chunk {
    fn clone(&self) -> Chunk {
        Chunk {
            data: self.data,
            lowest_x: 0,
            highest_x: 0,
            lowest_y: 0,
            highest_y: 0
        }
    }
}

impl Copy for Chunk {

}

impl Chunk {
    pub fn new() -> Chunk {
        Chunk {
            data: 0,
            lowest_x: 0,
            highest_x: 0,
            lowest_y: 0,
            highest_y: 0

        }
    }

    // x -> 0..7, y -> 0..7
    pub fn flip_cell(&mut self, x: i32, y: i32) {
        let xor_bit = 1 << y * 8 << x;
        self.data ^= xor_bit;
        if self.get_cell(x, y) {
            self.update_bounds(x as u8, y as u8);
        }
    }

    pub fn set_cell(&mut self, x: i32, y: i32, val: bool) {
        let temp_bit = 1 << y * 8 << x;
        if val {
            // Set cell to true -> bitwise or
            self.data |= temp_bit;
            self.update_bounds(x as u8, y as u8);
        } else {
            // Set cell to false -> bitwise and with !temp_bit
            self.data &= !temp_bit;
        }
    }

    fn update_bounds(&mut self, x: u8, y: u8) {
        if x < self.lowest_x {
            self.lowest_x = x;
        } 
        if x > self.highest_x {
            self.highest_x = x;
        }

        if y < self.lowest_y {
            self.lowest_y = y;
        } 
        if y > self.highest_y {
            self.highest_y = y;
        }
    }

    pub fn get_row(&self, y: i32) -> u8 {
        (self.data >> y * 8) as u8
    }

    pub fn get_cell(&self, x: i32, y: i32) -> bool {
        let temp_bit = self.data >> y * 8 >> x;
        return (temp_bit & 0b1) == 0b1;
    }

    pub fn get_raw(&self) -> u64 {
        self.data
    }
}
