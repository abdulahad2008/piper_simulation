# hrg_human — attribution

Source: https://github.com/TUMcps/human-robot-gym, path `human_robot_gym/models/assets/human/`
(human.xml, meshes/*.stl) and `models/assets/textures/{skin,jeans,green-shirt}.png`.
Human-Robot Gym declares "MIT License" in its package metadata (pyproject.toml) and ships no LICENSE file.
Its README states the human model is taken from https://github.com/KlabCMU/kin-poly (BSD-3-Clause),
whose meshes are segments of the SMPL neutral body (https://smpl.is.tue.mpg.de, non-commercial licence).

Used here for figure rendering only. Changes made: texture paths, debug sites hidden, root body
rotated to z-up, mocap flag removed. If you redistribute commercially, replace the meshes.
