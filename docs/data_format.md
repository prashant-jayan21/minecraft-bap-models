# Data format
Each sample contains the ground truth Builder action sequence, data which can be used as model input and some auxiliary data for convenience. It is a dict with the following keys:

- `builder_action_history`
  - The sequence of all previous Builder actions -- last element being the latest action
  - Datatype: List of `BuilderAction` objects. Each `BuilderAction` object has the following attributes:
    - `"action_type"`: string -- `"placement"` or `"removal"`
    - `"block"`: dict as follows:
      - `"x"`: the x-coordinate as an int
      - `"y"`: the y-coordinate as an int
      - `"z"`: the z-coordinate as an int
      - `"type"`: the color as a string (`"red"`, `"blue"`, etc.)
    
- `next_builder_actions`
  - The ground truth sequence of Builder actions -- this is the prediction goal
  - Datatype: Same as above

- `prev_utterances`
  - The sequence of all previous utterances in the dialog history -- last element being the latest utterance
  - Datatype: List of dicts. Each dict contains the following:
    - `"speaker"`: `"Builder"` or `"Architect"`
    - `"utterance"`: List of string tokens in the utterance

- `gold_config`
  - The target structure (This should not be accessed for modeling purposes as the Builder does not have access to the target structure during the game. We provide it for general convenience.)
  - Datatype: List of dicts. Each dict represents a block as mentioned above when describing `builder_action_history`.

- `built_config`
  - The built structure after the ground truth Builder actions take place
  - Datatype: Same as above

- `prev_config`
  - The built structure before the ground truth Builder actions take place
  - Datatype: Same as above

- `prev_builder_position`
  - The position of the Builder when the last utterance came in
  - Datatype: A dict containing the following:
    - `"x"`: the x-coordinate as a float
    - `"y"`: the y-coordinate as a float
    - `"z"`: the z-coordinate as a float
    - `"yaw"`: the yaw angle as a float
    - `"pitch"`: the pitch angle as a float

- `perspective_coordinates`
  - The perspective coordinates for the entire 3D grid wrt `prev_builder_position`
  - Datatype: A 3x11x9x11 PyTorch tensor of floats consisting of the 3 (x, y, z) perspective coordinates for every cell in the 11x9x11 3D grid (ordered x, y, z)

- `from_aug_data`
  - Whether or not this sample came from the synthetic data
  - Datatype: bool

- `json_id`
  - The index of the game log in the list of game logs stored in the `*jsons.pkl` file
  - Datatype: int

- `sample_id`
  - The index of the game state in the list of game states within the game log (this is the list stored at the `WorldStates` key in the game log)
  - Datatype: int

- `orig_experiment_id`
  - original log directory (also an experiment ID) for the game log in `data/logs/*/logs/` (for synthetic data -- this is the same as the original counterpart)
  - Datatype: string
