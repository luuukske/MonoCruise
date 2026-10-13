# Brief: independent safety and comfort review of AEB road-bent trajectories

You are reviewing an automatic emergency braking (AEB) system for trucks in
Euro Truck Simulator 2 / American Truck Simulator. You did not build the change.
Judge it on the evidence in each clip folder, not on what anyone hopes.

## The change

AEB predicts where the truck (ego) and every other vehicle will go, and brakes
when a predicted collision is close. Today the prediction is a constant-curvature
arc from the current steering (ego) or yaw rate (others). The new variant
("road-bent") uses the game's lane map as well. While ego is following its lane
(on it, roughly aligned, turning within the road's curvature), AEB also builds a
road path: it starts on the current steering and blends into the lane's curvature
over about a second of travel. A corner too tight for the speed is ignored from
where it starts. Every vehicle is still judged on today's arcs; only a vehicle
that sits in ego's lane ahead along the road path, but not along today's arc, is
judged on the road path instead (its own path bent along its lane too).

Expected upside: earlier braking for a stopped, crashed or much slower vehicle in
ego's lane round a bend, which today's arc only finds once ego has turned in.
Expected risk: braking added or advanced for a vehicle ego would not have reached,
e.g. when ego left the lane, changed lanes or the map is wrong.

## What each clip folder holds

- `summary.json`: session (SP = single player AI traffic, TMP = online players),
  ego speed, and per variant the first warn time, first brake time, and how many
  ticks it warned or braked.
- `tNN.NN.png`: two panels at the same moment, left today, right road-bent.
  Ego is at the origin facing up. Blue: ego's predicted corridor (today's arc).
  Yellow dashed: the road path (right panel only, when ego follows its lane). Red outline:
  a vehicle the variant flagged as colliding. Green: other vehicles. Labels give
  id and km/h. White: where ego actually drove afterwards. Grey: lane edges
  (4.5 m lanes); dotted: lane centres. The title says whether the variant was
  quiet, warning or braking, and its required deceleration.
- `thumb.jpg`, when present: a small screenshot of the game at the trigger.

## What to decide per clip

Compare the white line (what really happened) with the vehicle bodies:

1. **Safety regression**: road-bent removes or delays braking for a vehicle
   that ego actually hit or passed within about 0.5 m, or that sat in ego's
   lane on ego's real path.
2. **Safety improvement**: road-bent brakes earlier, or only road-bent brakes,
   for a vehicle that really was in ego's way.
3. **Comfort improvement**: road-bent removes or shortens braking or warning
   for a vehicle that ego passed clear of (about 1 m or more) and that was
   never in ego's way.
4. **Comfort regression**: road-bent adds or lengthens braking or warning for a
   vehicle that ego passed clear of.
5. **Neutral / unclear**: little difference, or the evidence does not show.

Be strict on safety: when unsure whether a removed brake mattered, say so and
mark it unclear rather than improvement. A simulator is also a game: players
sometimes drive into things on purpose or ghost through each other online.

## Output

Append one JSON object per clip, one per line, to the results file you are
given, with exactly these keys:

```json
{"clip": "60ff2992", "category": "comfort_improvement", "safety_risk": "none",
 "preferred": "road_bent", "confidence": 0.8,
 "rationale": "TMP truck parked off the outside of a left bend; ego followed the bend 4 m clear. Today braked at 8 m/s2 for it; road-bent stayed quiet."}
```

- `category`: `safety_regression`, `safety_improvement`, `comfort_improvement`,
  `comfort_regression`, `neutral` or `unclear`.
- `safety_risk` of the road-bent behaviour in this clip: `none`, `low`,
  `medium` or `high`.
- `preferred`: `today`, `road_bent` or `either`.
- `confidence`: 0 to 1.
- `rationale`: one or two sentences naming the vehicle id and what you saw.
