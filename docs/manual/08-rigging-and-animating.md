# Rigging and animating

A rig is a set of joints that a model hangs on, so that turning a joint bends the model. In Clay you build one the
way you built the model: from the same shapes, by hand, with nothing to fit and nothing to wait for. This
tutorial takes a figure made of boxes from "no bones" to a walk cycle in a game-ready GLB.

It needs no GPU and no weights. Only the last, optional step needs Blender.

## Build the figure

Start a document (`Ctrl+N`) and block out a figure from boxes: a torso, a head, two arms and two legs. Use the
rail's **+** to place boxes and `G`, `R` and `S` to move, turn and size them, as in [Modelling](07-modelling.md).
Give each part a name in the outliner: `Torso`, `Head`, `ArmBox.L`, `ArmBox.R`, `LegBox.L`, `LegBox.R`. The boxes
are only shapes. The names an imported animation is matched on later belong to the *bones*, which you add and name
in the next step, so a box keeps a name of its own (here with `Box` in it) rather than the bone's.

Nothing is a bone yet. A bone is an ordinary shape that you have *marked*, and the choice is yours, which is why
you can rig a model of any shape.

## Add the joints

Select a shape and choose **Mark as Bone** in the right-click menu's **Rig** group (or `Ctrl`+`K`, then type
*bone*). Its origin is now the joint's pivot, and a small cross with a line to its parent shows in the viewport. A
joint that is only a joint, with no shape of its own, is **Add Bone**: with one bone selected it adds the next
one a step above it, so a chain grows upward like a spine.

For the figure, mark the torso as the root (it is a bone named `Torso`) and add a bone for each limb at the point it
bends: a shoulder, a hip. Name the new bones `UpperArm.L`, `UpperArm.R`, `Leg.L` and `Leg.R`: a bone's name is what an
imported animation is matched on, and Clay reads `.L`, `_L` and `Left` alike. Move each into place with `G`; the
pivot is the joint, so a shoulder bone belongs at the shoulder and not in the middle of the arm.

Then make the hierarchy. Drag a row in the outliner onto another to make it a child, or select shapes and use
**Parent to Last**. Put the arm and leg bones under the torso, and the shapes under their bones: `ArmBox.L` under
the `UpperArm.L` bone, `LegBox.L` under the `Leg.L` bone. A shape that sits under a bone moves with it, however
many groups down, and nothing else needs doing. That is the whole of the binding for a model built from boxes.

## Pose it

Press `5`, or choose **Pose** in the header's mode switch. The model looks the same, but the viewport now shows
the *pose*, bones are what you pick, and the timeline opens under it.

Click an arm (or its bone's cross) and turn it with `R`. The whole arm swings from the shoulder, and the box
under it with it. Nothing about the model's real shape has changed: the rest pose is still what you built, and
leaving Pose mode brings it back.

With **Auto-key** on, which it is, turning the arm has also made a **key** at the playhead: a stored rotation for
that joint at that time. There was no animation, so one was made.

## Make it move

Make a walk in four poses.

1. With the playhead at 0, select every bone (`Ctrl`+`A`) and press `I` to key the figure as it stands: the
   start of the stride.
2. Move the playhead with the ruler, `Right`, or by dragging it, to about half a second. Turn the left arm
   forward and the right arm back, and the legs the other way: each turn is keyed as you let go.
3. Go to a second, and turn the arms and legs back the other way.
4. Go to one and a half seconds, and give it the pose it started in. Copying it exactly makes the loop seamless.

Press `Space`. The figure walks, round and round, because **Loop** is on. Each diamond in the dope sheet is a key;
drag one to retime it, `Delete` removes the ones you have picked, and a second animation is a second **New**.
Scrubbing, playing and stepping are never undo steps; every key you make is.

## Take it out

**Export GLB...** writes the rig and the animation as one file that an engine reads: the bones, the shapes bound
to them and the walk. **Export to the library** keeps it as an asset, and **Export Posed GLB...** writes one frame
of it. An engine that reads glTF animations gets the walk as a named clip.

A smooth model, rather than a model of boxes, is the case the second binding is for: with bones in place, **Weight
with Blender...** on the Inspector's **Bone & rig** card asks Blender to work out how much each point follows each
bone, so an elbow bends instead of hinging. That is the one step in this tutorial that needs Blender, and
[Clay](27-clay.md#rigging-with-blender-weights) describes it.

## Try it

1. Build the figure, rig it and make the four-pose walk above.
2. Add a second animation, **Idle**, with the arms moving a little, and switch between the two in the timeline's
   animation list.
3. Export the GLB and import it into an engine (see [Putting it in a game](12-putting-it-in-a-game.md)) to see
   both clips.

## What to read next

[Clay](27-clay.md#bones-and-rigs) has the bones, Pose mode, the timeline and the imports (an animation from another
file, matched to your bones by name). [Building a map](09-building-a-map.md) is the next tutorial, and
[Putting it in a game](12-putting-it-in-a-game.md) is where a file goes after that.
