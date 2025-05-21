# rust-tui-games

This project is a collection of Rust crates designed to create a terminal-based game using Q-learning for AI behavior. The project is structured into four main crates:

## Crates Overview

- **game**: This crate contains the core game logic, including the game state management and action handling.
- **env**: This crate provides environment-related functionality that can be utilized by both the game and machine learning components.
- **machine_learning**: This crate implements Q-learning algorithms, allowing the game to learn optimal actions through reinforcement learning.
- **playground**: This is the binary crate that ties together the functionality of the other three crates, serving as the entry point for running the game.

## Project Structure

```
.
├── Cargo.toml
├── README.md
├── env
│   ├── Cargo.toml
│   └── src
│       └── lib.rs
├── games
│   ├── Cargo.toml
│   └── src
│       ├── lib.rs
│       ├── maze
│       │   ├── game.rs
│       │   └── map.rs
│       └── maze.rs
├── machine_learning
│   ├── Cargo.toml
│   └── src
│       ├── lib.rs
│       ├── q_learning
│       │   ├── q_learning.rs
│       │   └── q_utils.rs
│       └── q_learning.rs
└── playground
    ├── Cargo.toml
    └── src
        └── main.rs
```

## Getting Started

To get started with the project, clone the repository and navigate to the project directory. You can build and run the playground crate, which will execute the game.

```bash
cargo run -p playground
```

## Dependencies

Each crate has its own dependencies specified in their respective `Cargo.toml` files. The root `Cargo.toml` file manages the workspace and includes all the crates as members.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.