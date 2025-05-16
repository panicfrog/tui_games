// maze.rs
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Clone, Copy)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum TileType {
    Wall,
    Path,
    Exit,
}
pub struct Maze {
    pub width: usize,
    pub height: usize,
    pub grid: Vec<Vec<TileType>>,
    pub start: Position,
    #[allow(dead_code)]
    pub goal: Position,
}

impl Maze {
    pub fn new(width: usize, height: usize) -> Self {
        let width = if width % 2 == 0 { width - 1 } else { width };
        let height = if height % 2 == 0 { height - 1 } else { height };

        let mut grid = vec![vec![TileType::Wall; width]; height];
        let mut rng = rand::rng();
        let mut walls = Vec::new();

        let sx = rng.random_range(1..width / 2) * 2 - 1;
        let sy = rng.random_range(1..height / 2) * 2 - 1;

        grid[sy][sx] = TileType::Path;
        for (dx, dy) in &[(2, 0), (-2, 0), (0, 2), (0, -2)] {
            let nx = (sx as isize + dx) as usize;
            let ny = (sy as isize + dy) as usize;
            if nx < width && ny < height {
                walls.push((nx, ny, sx, sy));
            }
        }

        while let Some((wx, wy, px, py)) = walls.pop() {
            if grid[wy][wx] == TileType::Wall {
                let dx = (wx + px) / 2;
                let dy = (wy + py) / 2;
                grid[wy][wx] = TileType::Path;
                grid[dy][dx] = TileType::Path;

                for (ox, oy) in &[(2, 0), (-2, 0), (0, 2), (0, -2)] {
                    let nx = wx as isize + ox;
                    let ny = wy as isize + oy;
                    if nx > 0 && nx < (width as isize) && ny > 0 && ny < (height as isize) {
                        if grid[ny as usize][nx as usize] == TileType::Wall {
                            walls.push((nx as usize, ny as usize, wx, wy));
                        }
                    }
                }

                walls.shuffle(&mut rng);
            }
        }

        let start = Position { x: 1, y: 1 };
        let goal = Position {
            x: width - 2,
            y: height - 2,
        };
        grid[goal.y][goal.x] = TileType::Exit;

        Maze {
            width,
            height,
            grid,
            start,
            goal,
        }
    }
}
