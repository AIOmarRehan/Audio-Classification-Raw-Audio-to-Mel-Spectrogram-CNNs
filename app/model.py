import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("saved_model/Audio_Model_Classification.h5")

# IMPORTANT: Keep same order as training (alphabetical default in TF)
CLASS_NAMES = ["Baby Cry", "Chainsaw", "Clock Tick", "Cow", "Dog", "Fire Crackling", "Frog", "Helicopter", "Person Sneeze", "Pig", "Rain", "Rooster", "Sea Waves"]

def predict(img):
    # Convert PIL image → numpy array
    img = np.array(img).astype("float32") / 255.0  # normalize [0,1]

    # Resize to match training target (231x232)
    img = tf.image.resize(img, (231, 232))  # (231, 232, 4)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # (1, 231, 232, 4)

    # Predict
    preds = model.predict(img)
    probs = preds[0]

    class_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return CLASS_NAMES[class_idx], confidence, prob_dict

# THIS OUR PAST VERSION OF... IT WAS OUT RESIZING BUT EVEN THOUGH THE MODEL WAS WORKING FINE, SO, MAYBE TF COULD ACCEPT DYNAMIC SIZES OF INPUT IMAGES
# def predict(img):
#     # Convert to numpy array (RGBA)
#     img = np.array(img) / 255.0  # shape (H, W, 4)
#     img = np.expand_dims(img, axis=0)  # (1, H, W, 4)
#
#     # Predict
#     preds = model.predict(img)
#     probs = preds[0]
#
#     class_idx = int(np.argmax(probs))
#     confidence = float(np.max(probs))
#     prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
#
#     return CLASS_NAMES[class_idx], confidence, prob_dict