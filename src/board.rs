use std::{collections::HashMap, fs::File, io::{BufReader, Read, Write}};

pub struct Board {
    pub chunks: HashMap<(i32, i32), Chunk>,
}

impl Board {
    pub fn new() -> Board {
        Board {
            chunks: HashMap::new()
        }
    }

    pub fn load(&mut self, file: &str) {
        let file_o = File::open(file).unwrap();
        let mut buf_reader = BufReader::new(file_o);
        let mut len_buf = [0u8; 8];
        buf_reader.read_exact(&mut len_buf).unwrap();
        let len = usize::from_be_bytes(len_buf);
        self.chunks = HashMap::new();
        
        for _i in 0..len {
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
    }

    pub fn save(&self, file: &str) {
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
            chunks: self.chunks.clone()
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn flip_cell(&mut self, x: i32, y: i32) {
        let (cx, cy, ix, iy) = self.get_chunk_coords(x, y);
        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.flip_cell(ix, iy);
        if chunk.data == 0 {
            self.chunks.remove(&(cx, cy));
        }
    }

    pub fn set_cell(&mut self, x: i32, y: i32, state: bool) {
        let (cx, cy, ix, iy) = self.get_chunk_coords(x, y);
        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.set_cell(ix, iy, state);
        if !state && chunk.data == 0 {
            self.chunks.remove(&(cx, cy));
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

    pub fn step_simulation(&mut self) {
        //let next_board = self.clone();
        let (array, to_check, x_offset, y_offset, width, height) = self.compress_chunk_array();
        let mut true_cells = vec![];

        // TODO: Apply game logic inside of modified_array using to_check coordinates
        for (x, y) in to_check {
            let val = array[y][x];
            let mut neighbor_count = 0;

            for i in 0..=2 {
                for j in 0..=2 {
                    if i == 1 && j == 1 {
                        continue;
                    }

                    let nx = i32::try_from(x).unwrap() + i32::try_from(i).unwrap() - 1;
                    let ny = i32::try_from(y).unwrap() + i32::try_from(j).unwrap() - 1;

                    if nx < 0 || ny < 0 || nx > width - 1 || ny > height - 1 {
                        continue;
                    }

                    if array[usize::try_from(ny).unwrap()][usize::try_from(nx).unwrap()] {
                        neighbor_count += 1;
                    }
                }
            }

            /*if neighbor_count != 0 {
                println!("{}", neighbor_count);
            }*/

            if val {
                match neighbor_count {
                    ..=1 => {},
                    2..=3 => {
                        true_cells.push((x, y));
                    },
                    4.. => {}
                }
            } else {
                if neighbor_count == 3 {
                    true_cells.push((x, y));
                }
            }
        }

        self.chunks = self.decompress_chunk_array(true_cells, x_offset, y_offset);
    }

    pub fn decompress_chunk_array(&self, true_cells: Vec<(usize, usize)>, x_offset: i32, y_offset: i32) -> HashMap<(i32, i32), Chunk> {
        let mut new_board = Board::new();

        for (ax, ay) in true_cells {
            let gx = i32::try_from(ax).unwrap() - x_offset;
            let gy = i32::try_from(ay).unwrap() - y_offset;

            new_board.set_cell(gx, gy, true);
        }

        return new_board.chunks;
    }

    pub fn compress_chunk_array(&self) -> (Vec<Vec<bool>>, Vec<(usize, usize)>, i32, i32, i32, i32) {
        // Generate a 2d array of bools for the current board state, padded by 1 around every edge
        // for quick calculations
        
        let (lx, ux, ly, uy) = self.get_bounds();
        let width = (ux - lx) + 3;
        let height = (uy - ly) + 3;
        //println!("Bounds: X-({} - {}), Y-({} - {})", lx, ux, ly, uy);

        let conversion_x = -lx + 1;
        let conversion_y = -ly + 1;

        let mut array = vec![vec![false; width as usize]; height as usize];

        let mut to_check = vec![];

        // Fill in the actual chunk data
        for (chunk_coords, chunk) in &self.chunks {
            for ix in 0..8 {
                for iy in 0..8 {
                    let (global_x, global_y) = self.get_global_coords(chunk_coords.0, chunk_coords.1, ix, iy);
                    
                    //println!("{} - {}, {} - {}", global_x, conversion_x, global_y, conversion_y);
                    
                    let ax_i = global_x + conversion_x;
                    let ay_i = global_y + conversion_y;

                    if ax_i < 0 || ay_i < 0 {
                        continue;
                    }

                    let ax: usize = ax_i.try_into().unwrap();
                    let ay: usize = ay_i.try_into().unwrap();

                    if ax >= width.try_into().unwrap() || ay >= height.try_into().unwrap() {
                        continue;
                    }

                    if chunk.get_cell(ix, iy) {
                        for i in 0..=2 {
                            for j in 0..=2 {
                                let coords = (ax + i - 1, ay + j - 1);
                                if !to_check.contains(&coords) {
                                    to_check.push(coords);
                                }
                            }
                        }
                    }

                    array[ay][ax] = chunk.get_cell(ix, iy);
                }
            }
        }

        return (array, to_check, conversion_x, conversion_y, width, height);
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
