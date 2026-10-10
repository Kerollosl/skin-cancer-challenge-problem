"""Train on training data; evaluate an untouched, deterministic test generator."""
import argparse,json
from pathlib import Path
import numpy as np
from evaluation import metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,default=Path("skin_cancer_dataset"))
    ap.add_argument("--architecture",choices=["VGG16","InceptionV3","tuned"],default="VGG16")
    ap.add_argument("--epochs",type=int,default=3);ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--output",type=Path,default=Path("artifacts"));ap.add_argument("--threshold",type=float,default=0.5)
    args=ap.parse_args()
    if args.epochs<1 or not 0<=args.threshold<=1:ap.error("Positive epochs and a threshold in [0,1] are required")
    import tensorflow as tf
    import pandas as pd
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    tf.keras.utils.set_random_seed(args.seed)
    size=(75,75)
    if args.architecture=="VGG16":
        from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
        base=VGG16(include_top=False,weights="imagenet",input_shape=(*size,3))
    elif args.architecture=="InceptionV3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
        base=InceptionV3(include_top=False,weights="imagenet",input_shape=(*size,3))
    else:preprocess_input=lambda x:x/255.0;base=None
    common=dict(target_size=size,batch_size=64,class_mode="binary",classes=["Benign","Malignant"],seed=args.seed)
    generator=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)
    train=generator.flow_from_directory(str(args.data/"train"),subset="training",shuffle=True,**common)
    val=generator.flow_from_directory(str(args.data/"train"),subset="validation",shuffle=False,**common)
    test=ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(str(args.data/"test"),shuffle=False,**common)
    if not min(train.samples,val.samples,test.samples):raise ValueError("Every split must contain images")
    if base is None:
        from keras_tuner_script import tune_hyperparameters
        model=tune_hyperparameters(train,val)
    else:
        base.trainable=False
        model=tf.keras.Sequential([base,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dropout(.25),tf.keras.layers.Dense(1,activation="sigmoid")])
        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        model.fit(train,validation_data=val,epochs=args.epochs)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,g in [("validation",val),("test",test)]:
        g.reset();scores=model.predict(g,verbose=0).reshape(-1)
        pd.DataFrame({"Image_Name":g.filenames,"Prediction Value":scores,"Actual":np.where(g.classes==1,"Malignant","Benign")}).to_csv(args.output/f"{name}_preds.csv",index=False)
        (args.output/f"{name}_metrics.json").write_text(json.dumps(metrics(g.classes,scores,args.threshold),indent=2))
    model.save(args.output/"model.keras")
    (args.output/"metadata.json").write_text(json.dumps({"architecture":args.architecture,"size":size,"classes":test.class_indices,"seed":args.seed,"threshold":args.threshold},indent=2))
    print((args.output/"test_metrics.json").read_text())
if __name__=="__main__":main()
