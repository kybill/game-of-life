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

    pub fn flip_cell(&mut self, x: i32, y: i32) {
        let (cx, cy, ix, iy) = self.get_chunk_coords(x, y);
    }

    pub fn get_or_create_chunk(&mut self, chunk_x: i32, chunk_y: i32) -> &mut Chunk { 
        let chunk = self.get_chunk(chunk_x, chunk_y);

        if chunk.is_some() {
            return chunk.unwrap();
        } else {
            let mut new_chunk = Chunk::new();
            self.chunks.insert((chunk_x, chunk_y), new_chunk);
        }

        self.chunks.get_mut(&(chunk_x, chunk_y)).unwrap()
    }

    pub fn get_chunk(&mut self, chunk_x: i32, chunk_y: i32) -> Option<&mut Chunk> {
        self.chunks.get_mut(&(chunk_x, chunk_y))
    }

    pub fn get_chunk_coords(&self, x: i32, y: i32) -> (i32, i32, i32, i32) { // chunk_x, chunk_y, internal_x, internal_y
        let chunk_x = if x >= 0 { x / 8 } else { (x + 1) / 8 - 1 };
        let chunk_y = if y >= 0 { y / 8 } else { (y + 1) / 8 - 1 };

        let internal_x = if x >= 0 { x % 8 } else { (x+1) % 8 };
        let internal_y = if y >= 0 { y % 8 } else { (y+1) % 8 };

        //println!("{} - {} - {} - {}", chunk_x, chunk_y, internal_x, internal_y);

        (chunk_x, chunk_y, internal_x, internal_y)
    }
}

pub struct Chunk {
    data: u64
}

impl Chunk {
    pub fn new() -> Chunk {
        Chunk {
            data: 0
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
        } else {
            // Set cell to false -> bitwise and with !temp_bit
            self.data &= !temp_bit;
        }
    }

    pub fn get_row(&self, y: i32) -> u8 {
        (self.data >> y * 8) as u8
    }

    pub fn get_raw(&self) -> u64 {
        self.data
    }
}
