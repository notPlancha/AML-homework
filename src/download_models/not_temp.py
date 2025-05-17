import os
from pyprojroot.here import here

os.environ["KERAS_BACKEND"] = "torch"
# os.environ["HF_HOME"] = str(here("cache"))
import keras
model = keras.saving.load_model("hf://apple/mobilevit-xx-small")

# model = keras.saving.load_model("hf://fahd9999/face_shape_classification")
print(model.summary())

"""
DOESNT WORK IDK WHY TRY INSTEAD TO DOWNLAOD MANUALLY AND USE THIS TO CHANGE IT TO A .KERAS MODEL :))))))))))))))
"""