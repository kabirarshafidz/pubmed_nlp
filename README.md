# PubMed NLP Analysis

## Dataset
# Data Loading
# Data Preprocessing
### Preparing the data
### Preparing the data
### Preparing the data
### Preparing the data
### Preparing the data

## Models Used
# Model Training
- model 0: Naive bayes with TF-IDF encoder (baseline)
- model 1: Conv1D with token embeddings
- model 2: TensorFlow Hub Pretrained Feature Extractor
- model 3: Conv1D with character embeddings
- model 4: Pretrained token embeddings (same as 2) + character embeddings (same as 3)
- model 5: Pretrained token embeddings + character embeddings + positional embeddings
- model 6: Pretrained token embeddings + character embeddings + relative positional embeddings
## Model 0
Naive bayes with TF-IDF encoder (baseline)
from sklearn.feature_extraction.text import tfidfvectorizer
from sklearn.naive_bayes import multinomialnb
from sklearn.pipeline import pipeline

model_0 = pipeline([
  ("tfidf", tfidfvectorizer()),
  ("clf", multinomialnb())
])

model_0.fit(train_sentences, train_labels_encoded)
## Model 1
Conv1D with token embeddings
### Model building and training
history_1 = model_1.fit(train_dataset,
                        epochs=5,
                        steps_per_epoch=int(0.1*len(train_dataset)),
                        validation_data=val_dataset,
                        validation_steps=int(0.1*len(val_dataset)))
## Model 2
TensorFlow Hub Pretrained Feature Extractor
### Model building and training
history_2 = model_2.fit(train_dataset,
                        epochs=5,
                        steps_per_epoch=int(0.1*len(train_dataset)),
                        validation_data=val_dataset,
                        validation_steps=int(0.1*len(val_dataset)))
## Model 3
Conv1D with character embeddings
### Model building and training
history_3 = model_3.fit(train_char_dataset,
                        epochs=5,
                        steps_per_epoch=int(0.1*len(train_char_dataset)),
                        validation_data=val_char_dataset,
                        validation_steps=int(0.1*len(val_char_dataset)))
## Model 4
Pretrained token embeddings (same as 2) + character embeddings (same as 3)
### Model building and training
history_4 = model_4.fit(train_combined_dataset,
                        epochs=5,
                        steps_per_epoch=int(0.1*len(train_combined_dataset)),
                        validation_data=val_combined_dataset,
                        validation_steps=int(0.1*len(val_combined_dataset)))
## Model 5
Pretrained token embeddings + character embeddings + positional embeddings*

*positional embeddings: which part of the abstract the text is located.
### Model building and training
history_5 = model_5.fit(train_char_token_pos_dataset,
                        epochs=5,
                        steps_per_epoch=int(0.1*len(train_char_token_pos_dataset)),
                        validation_data=val_char_token_pos_dataset,
                        validation_steps=int(0.1*len(val_char_token_pos_dataset)))
## Model 6
Pretrained token embeddings + character embeddings + relative positional embeddings*

* relative position: line no / total line no
### Model building and training
history_6 = model_6.fit(train_char_token_relpos_dataset,
                        epochs=5,
                        steps_per_epoch=int(0.1*len(train_char_token_relpos_dataset)),
                        validation_data=val_char_token_relpos_dataset,
                        validation_steps=int(0.1*len(val_char_token_relpos_dataset)))
The table and the bar chart shows that model_6 is the best performing model.
# Saving the best model

## Results
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  calculates model accuracy, precision, recall and f1 score of a binary classification model.

  args:
      y_true: true labels in the form of a 1d array
      y_pred: predicted labels in the form of a 1d array

  returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
evaluation_0 = model_0.score(val_sentences, val_labels_encoded)
results_0 = calculate_results(test_labels_encoded, baseline_preds)
results_0
from tf_keras import layers
input = layers.input((1,), dtype=tf.string)
x = text_vectorizer(input)
x = embedding(x)
x = layers.conv1d(64, kernel_size=5, padding='same', activation='relu')(x)
x = layers.globalaveragepooling1d()(x)
output = layers.dense(num_classes, activation='softmax')(x)
model_1 = keras.model(input, output)

model_1.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.adam(),
                metrics=['accuracy'])
model_1_pred_probs = model_1.predict(test_dataset)
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)
results_1 = calculate_results(test_labels_encoded, model_1_preds)
results_1
results_0
model_2 = keras.sequential([
    sentence_encoder_layer,
    layers.dense(128, activation='relu'),
    layers.dense(num_classes, activation='softmax')
])

model_2.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.adam(),
                metrics=['accuracy'])
model_2_pred_probs = model_2.predict(test_dataset)
model_2_preds = tf.argmax(model_2_pred_probs, axis=1)
results_2 = calculate_results(test_labels_encoded, model_2_preds)
results_2
results_0
input = layers.input((1,), dtype=tf.string)
x = char_vectorizer(input)
x = char_embedding(x)
x = layers.conv1d(64, kernel_size=5, padding='same', activation='relu')(x)
x = layers.globalaveragepooling1d()(x)
output = layers.dense(num_classes, activation='softmax')(x)
model_3 = keras.model(input, output)

model_3.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.adam(),
                metrics=['accuracy'])
model_3_pred_probs = model_3.predict(test_char_dataset)
model_3_preds = tf.argmax(model_3_pred_probs, axis=1)
results_3 = calculate_results(test_labels_encoded, model_3_preds)
results_3
results_0
# token model
token_input = layers.input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = sentence_encoder_layer(token_input)
token_output = layers.dense(128, activation="relu")(token_embeddings)
token_model = keras.model(token_input, token_output)

# char model
char_input = layers.input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_input)
char_embeddings = char_embedding(char_vectors)
char_bi_lstm = layers.bidirectional(layers.lstm(25))(char_embeddings) # bi-lstm shown in figure 1 of https://arxiv.org/pdf/1612.05251.pdf
char_model = keras.model(char_input, char_bi_lstm)

# concatenate
concat = layers.concatenate()([token_model.output, char_model.output])

# output layer
combined_dropout = layers.dropout(0.5)(concat)
combined_dense = layers.dense(128, activation='relu')(combined_dropout)
final_dropout = layers.dropout(0.5)(combined_dense)
output_layer = layers.dense(num_classes, activation='softmax')(final_dropout)

# construct model
model_4 = keras.model([token_model.input, char_model.input], output_layer,
                      name='token_char_model')

model_4.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.adam(),
                metrics=['accuracy'])

model_4.summary()
model_4_pred_probs = model_4.predict(test_combined_dataset)
model_4_preds = tf.argmax(model_4_pred_probs, axis=1)
results_4 = calculate_results(test_labels_encoded, model_4_preds)
results_4
results_0
# token model
token_input = layers.input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = sentence_encoder_layer(token_input)
token_output = layers.dense(128, activation="relu")(token_embeddings)
token_model = keras.model(token_input, token_output)

# char model
char_input = layers.input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_input)
char_embeddings = char_embedding(char_vectors)
char_bi_lstm = layers.bidirectional(layers.lstm(25))(char_embeddings) # bi-lstm shown in figure 1 of https://arxiv.org/pdf/1612.05251.pdf
char_model = keras.model(char_input, char_bi_lstm)

# line number model
line_number_input = layers.input(shape=(15,), dtype=tf.float32, name="line_number_input")
line_number_dense = layers.dense(32, activation='relu')(line_number_input)
line_number_model = keras.model(line_number_input, line_number_dense)

# total line model
total_line_input = layers.input(shape=(20,), dtype=tf.float32, name="total_line_input")
total_line_dense = layers.dense(32, activation='relu')(total_line_input)
total_line_model = keras.model(total_line_input, total_line_dense)

# concatenate
concat = layers.concatenate()([token_model.output, char_model.output])
concat = layers.dense(256, activation='relu')(concat)
concat = layers.dropout(0.5)(concat)

# tribrid concatenation
tribrid_concat = layers.concatenate()([line_number_model.output,
                                      total_line_model.output,
                                      concat])


# output layer
output_layer = layers.dense(num_classes, activation='softmax')(tribrid_concat)

# construct model
model_5 = keras.model([line_number_model.input,
                       total_line_model.input,
                       token_model.input,
                       char_model.input],
                      output_layer,
                      name='token_char_model')

model_5.compile(loss=keras.losses.categoricalcrossentropy(label_smoothing=0.2),
                optimizer=keras.optimizers.adam(),
                metrics=['accuracy'])

model_5.summary()
model_5_pred_probs = model_5.predict(test_char_token_pos_dataset)
model_5_preds = tf.argmax(model_5_pred_probs, axis=1)
results_5 = calculate_results(test_labels_encoded, model_5_preds)
results_5
results_0
# token model
token_input = layers.input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = sentence_encoder_layer(token_input)
token_output = layers.dense(128, activation="relu")(token_embeddings)
token_model = keras.model(token_input, token_output)

# char model
char_input = layers.input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_input)
char_embeddings = char_embedding(char_vectors)
char_bi_lstm = layers.bidirectional(layers.lstm(25))(char_embeddings) # bi-lstm shown in figure 1 of https://arxiv.org/pdf/1612.05251.pdf
char_model = keras.model(char_input, char_bi_lstm)

# line number model
relative_position_input = layers.input(shape=(1,), dtype=tf.float32, name="relative_position_input")
relative_position_dense = layers.dense(32, activation='relu')(relative_position_input)
relative_position_model = keras.model(relative_position_input, relative_position_dense)

# concatenate
concat = layers.concatenate()([token_model.output, char_model.output])
concat = layers.dense(256, activation='relu')(concat)
concat = layers.dropout(0.5)(concat)

# tribrid concatenation
tribrid_concat = layers.concatenate()([relative_position_model.output,
                                      concat])


# output layer
output_layer = layers.dense(num_classes, activation='softmax')(tribrid_concat)

# construct model
model_6 = keras.model([relative_position_model.input,
                       token_model.input,
                       char_model.input],
                      output_layer,
                      name='token_char_model')

model_6.compile(loss=keras.losses.categoricalcrossentropy(label_smoothing=0.2),
                optimizer=keras.optimizers.adam(),
                metrics=['accuracy'])

model_6.summary()
model_6_pred_probs = model_6.predict(test_char_token_relpos_dataset)
model_6_preds = tf.argmax(model_6_pred_probs, axis=1)
results_6 = calculate_results(test_labels_encoded, model_6_preds)
results_6
results_0
all_model_results = pd.dataframe({'model_0_baseline': results_0,
                                  'model_1_custom_token_embedding': results_1,
                                  'model_2_pretrained_token_embedding': results_2,
                                  'model_3_custom_char_embedding': results_3,
                                  'model_4_hybrid_char_token_embedding': results_4,
                                  'model_5_pos_char_token_embedding': results_5,
                                  'model_6_relative_position_embedding': results_6})

all_model_results = all_model_results.transpose()
all_model_results
all_model_results['accuracy'] = all_model_results['accuracy']/100

all_model_results.plot(kind='bar', figsize=(10,7)).legend(bbox_to_anchor=(1,1))
all_model_results.sort_values('f1', ascending=true)['f1'].plot(kind='bar', figsize=( 10,7))
loaded_pred_probs = loaded_model.predict(test_char_token_relpos_dataset)
loaded_preds = tf.argmax(loaded_pred_probs, axis=1)
loaded_results = calculate_results(test_labels_encoded, loaded_preds)
loaded_results

## How to Run
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
2. Open the Jupyter Notebook:  
   ```bash
   jupyter notebook pubmed_nlp.ipynb
   ```
3. Run all cells to reproduce results.

