// maze.rs
use env::rand::seq::SliceRandom;
use env::rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
pub struct MazeMap {
    pub width: usize,
    pub height: usize,
    pub grid: Vec<Vec<TileType>>,
    pub start: Position,
    #[allow(dead_code)]
    pub goal: Position,
}

impl MazeMap {
    pub fn new(width: usize, height: usize) -> Self {
        let width = if width % 2 == 0 { width - 1 } else { width };
        let height = if height % 2 == 0 { height - 1 } else { height };

        let mut grid = vec![vec![TileType::Wall; width]; height];
        let mut rng = env::rand::rng();
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

        MazeMap {
            width,
            height,
            grid,
            start,
            goal,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    pub fn opposite(&self) -> Action {
        match self {
            Action::Up => Action::Down,
            Action::Down => Action::Up,
            Action::Left => Action::Right,
            Action::Right => Action::Left,
        }
    }

    pub fn delta(&self) -> (isize, isize) {
        match self {
            Action::Up => (0, -1),
            Action::Down => (0, 1),
            Action::Left => (-1, 0),
            Action::Right => (1, 0),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MazeStatus {
    Ongoing,
    Win,
    FailDeadend,
    FailStepLimit,
}

impl MazeMap {
    pub fn max_steps(&self) -> usize {
        self.grid
            .iter()
            .flatten()
            .filter(|&&t| t != TileType::Wall)
            .count()
    }
    /// 返回允许的动作和当前状态
    pub fn allowed_actions_and_status(
        &self,
        pos: Position,
        last_action: Option<Action>,
        steps: usize,
    ) -> (Vec<Action>, MazeStatus) {
        let mut actions = Vec::new();
        for &action in &[Action::Up, Action::Down, Action::Left, Action::Right] {
            if let Some(last) = last_action {
                if action == last.opposite() {
                    continue;
                }
            }
            let (dx, dy) = action.delta();
            let nx = pos.x as isize + dx;
            let ny = pos.y as isize + dy;
            if nx >= 0
                && nx < self.width as isize
                && ny >= 0
                && ny < self.height as isize
                && (self.grid[ny as usize][nx as usize] == TileType::Path
                    || self.grid[ny as usize][nx as usize] == TileType::Exit)
            {
                actions.push(action);
            }
        }
        if self.grid[pos.y][pos.x] == TileType::Exit {
            (actions, MazeStatus::Win)
        } else if steps >= self.max_steps() {
            (actions, MazeStatus::FailStepLimit)
        } else if actions.is_empty() {
            (actions, MazeStatus::FailDeadend)
        } else {
            (actions, MazeStatus::Ongoing)
        }
    }
}
