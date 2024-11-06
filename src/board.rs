use std::collections::HashMap;

pub struct Board {
    chunks: HashMap<(i32, i32), Chunk>
}

impl Board {
    pub fn new() -> Board {
        Board {
            chunks: HashMap::new()
        }
    }

    pub fn clone(&self) -> Board {
        let mut keys = self.chunks.keys();
        let mut vals = self.chunks.values();

        let mut new_map = HashMap::new();

        let mut stop = false;
        while !stop {
            let key = keys.next();
            let val = vals.next();
            if key.is_none() || val.is_none() {
                stop = true;
                continue;
            }

            new_map.insert(*key.unwrap(), *val.unwrap());
        }

        Board {
            chunks: new_map
        }
    }

    pub fn flip_cell(&mut self, x: i32, y: i32) {
        let (cx, cy, ix, iy) = self.get_chunk_coords(x, y);
        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.flip_cell(ix, iy);
    }

    pub fn set_cell(&mut self, x: i32, y: i32, state: bool) {
        let (cx, cy, ix, iy) = self.get_chunk_coords(x, y);
        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.set_cell(ix, iy, state);
    }

    pub fn get_or_create_chunk(&mut self, chunk_x: i32, chunk_y: i32) -> &mut Chunk { 
        let chunk_map = &mut self.chunks;

        if chunk_map.contains_key(&(chunk_x, chunk_y)) {
            return chunk_map.get_mut(&(chunk_x, chunk_y)).unwrap();
        }

        let new_chunk = Chunk::new();
        chunk_map.insert((chunk_x, chunk_y), new_chunk);

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

        println!("{} - {}, {} - {} - {} - {}", x, y, chunk_x, chunk_y, internal_x, internal_y);
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
        let array = self.generate_array();
    }

    fn generate_array(&self) -> &[&[bool]] {
        // Generate a 2d array of bools for the current board state, padded by 1 around every edge
        // for quick calculations
        let mut array = vec![];

        let mut col = vec![];
        col.push(false);
        array.push(col.as_slice());

        return array.as_slice();
    }
}

pub struct Chunk {
    data: u64,
    lowest_x: u64,
    highest_x: u64,
    lowest_y: u64,
    highest_y: u64
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
    }

    pub fn set_cell(&mut self, x: i32, y: i32, val: bool) {
        let temp_bit = 1 << y * 8 << x;
        if val {
            // Set cell to true -> bitwise or
            self.data |= temp_bit;
            self.update_bounds(x as u64, y as u64);
        } else {
            // Set cell to false -> bitwise and with !temp_bit
            self.data &= !temp_bit;
        }
    }

    fn update_bounds(&mut self, x: u64, y: u64) {
        if x < self.lowest_x {
            self.lowest_x = x;
        } else if x > self.highest_x {
            self.highest_x = x;
        }

        if y < self.lowest_y {
            self.lowest_y = y;
        } else if y > self.highest_y {
            self.highest_y = y;
        }
    }

    pub fn get_row(&self, y: i32) -> u8 {
        (self.data >> y * 8) as u8
    }

    pub fn get_raw(&self) -> u64 {
        self.data
    }
}
