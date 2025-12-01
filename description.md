# Piano Hand Posture Analyzer

A piano hand posture classifier machine learning model with random forest algorithm. The system will have a video of the pianist playing as input, extract the needed features with python and media pipe hands and feed it to a random forest model to classify the hand posture as:

- Correct (Mid-range finger angles + Wrist Y-position < Knuckle Y-position) class0
- Flat Fingers (High finger angles) class1
- High Wrist (Wrist Y-position > Knuckle Y-position) class2
- Dropped Wrist (Wrist Y-position > Knuckle Y-position) class3
- Collapsed Finger Joints (Finger angles too low) class4

Model Features will be:

- Wrist-Knuckle Slope
- Finger Curvature Angles (Index, Middle, Pinky)
- DIP Angle (Index, Middle, Pinky)

Appropriate camera angle: Side view of the hands, high enough to see both hands and wrists clearly.

How will the video be treated? Separated in many spaced frames?

How much data do I need? And where can I find it? How should I label it?

I need scientific or bibliographic basis to be sure about the classes that i chose

Your Piano Technique is Holding You Back | Perfect Form Explained

The output for hands should be an array of classifications for each frame for each hand:
right_hand_classification = [(5, “Correct”), (10, “Correct”)]
left_hand_classification = [(5, “Flat Fingers”), (10, “Flat Fingers”)]

image for every 10% of the video
