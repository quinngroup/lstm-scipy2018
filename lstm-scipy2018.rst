:author: Christian McDaniel
:email: clm121@uga.edu
:institution: University of Georgia

:author: Shannon Quinn, PhD
:email: spq@uga.edu
:institution: University of Georgia
:bibliography: citations

--------------------------------------------------
Developing an LSTM Pipeline for Accelerometer data
--------------------------------------------------

.. class:: abstract

.. class:: keywords

Introduction
------------

Human Activity Recognition (HAR) is a time series classification problem in which a classifier attempts to discern distinguishable features from movement-capturing on-body sensors :cite:`KimCook2010`. Typical sensors record changes in velocity through time in the x- y- and z-directions, i.e., accelerometers. Accelerometer output consists of high-frequency (30-200Hz) triaxial time series recordings, often containing noise, imprecision, and missing data :cite:`Ravietal2005`, :cite:`BaoIntille2004`. Furthermore, periods of non-activity – commonly referred to as the Null class – typically represent a large majority of HAR-related accelerometer data :cite:`OrdonezRoggen2016`. Consequently, attempts to use traditional classifiers have resulted in an apparent upper bound on performance and typically require significant preprocessing and technical engineering of hand crafted features from raw data :cite:`Gaoetal2016`, :cite:`Charvarriagaetal2013`, :cite:`ManniniSabatini2010` :cite:`Gjoreskietal2016` :cite:`Ravietal2005` :cite:`OrdonezRoggen2016`.

The limitations of classical methods in this domain have been alleviated by concurrent theoretical and practical advancements in artificial neural networks, which are more suited for complex non-linear data. While convolutional neural networks (CNNs) are attractive for their automated feature extraction capabilities during convolution and pooling operations :cite:`Sjostrom2017` :cite:`Rassemetal2017` :cite:`Fiterauetal2016` :cite:`SeokKimPark2018` :cite:`Zebinetal2017` :cite:`Gaoetal2016` :cite:`Zhuetal2017` :cite:`OrdonezRoggen2016` :cite:`Gjoreskietal2016`, recurrent neural networks (RNNs) are specifically designed to extract information from time series data due to the recurrent nature of their data processing and weight updating operations :cite:`WilliamsZipser1989`. Furthermore, whereas earlier implementations of RNNs suffered from vanishing and exploding gradients during training, the incorporation of a multi-gated memory cell in long short-term memory recurrent neural networks (LSTMs) :cite:`HochreiterSchmidhuber1997` as well as regularization schemes such as an upper constraint known as gradient clipping :cite:`Pascanuetal2013` helped alleviate these issues.

Researchers in various fields have utilized these networks for their data modeling and analysis tasks :cite:`SukhwaniPanwar2016` :cite:`Wooetal2016` :cite:`Zhaoetal2016` :cite:`Gersetal2002` :cite:`Pigouetal2016` :cite:`Gravesetal2013`. As RNN usage increases, numerous studies have emerged to address various aspects of understanding and implementing these complex models, namely regarding the vast architectural and hyperparameter combinations that are possible :cite:`Gersetal2002` :cite:`ReimersGurevych2017` :cite:`PressWolf2017` :cite:`Karpathyetal2015` :cite:`Merityetal2017`. Unfortunately, these pioneering studies tend to focus on tasks other than HAR, leaving the time series classification tasks of HAR without domain-specific architecture guidance or insights into the models’ representation of the data. Consequently, as is discussed in detail below, pilot studies using LSTMs to classify accelerometer data have borrowed what they could from the use of LSTMs in other domains and used their best judgement for the remaining issues.

In a meta-analysis style overview of the use of LSTM RNNs for HAR experiments (discussed below), we found a general lack of consensus or justification regarding the various model architectures and hyperparameters used. Often, a given pair of experiments explored largely or entirely non-overlapping ranges of hyperparameter settings. For example, Carvalho, et. al. assessed the performance of models with two, five, or ten units per layer, Rassem, et. al. constructed models with 25, 35, or 50 units per layer, and Setterquist 2018 searched from 8-512 units per layer :cite:`Carvalhoetal2017` :cite:`Rassemetal2017` :cite:`Setterquist2018`. Furthermore, many architectural and procedural details are not included in reports, making reproducibility nearly impossible. The analysis pipelines employed are often vaguely described and significant sources of data leakage, where details from the testing data are exposed to the model during training, seem to be largely overlooked. Without clear justifications for model implementations and deliberate, reproducible data analysis pipelines, objective model comparisons and inferences from results cannot be made. For these reasons, the current report seeks to summarize the previous implementations of LSTMs for HAR research available in literature and outline a structured data analysis pipeline for this domain. We implement our pipeline, optimizing a baseline LSTM model over an expansive hyperparameter search space. We suspect that such efforts will provide unique insights into the usefulness of LSTMs for classifying accelerometer data and will allow for scientifically rigorous comparisons across experiments and datasets.

Related Works
-------------
The following section outlines the nuanced hyperparameter combinations used by 27 studies available in literature in a meta-analysis style survey. Published works as well as pre-published and thesis research projects were included so as to gain insight into the state-of-the-art methodologies at all levels and increase the volume of works available for review. It should be noted that the following summaries are not necessarily entirely exhaustive regarding the specifications listed or the individual citations made for each specification. Additionally, many reports did not include explicit details of many aspects of their research.

The survey of previous experiments in this field provided blueprints for constructing an adequate search space of hyperparameters. If the reader has a good understanding of the hyperparameters involved in training an LSTM model, he or she may choose to skip this section. Furthermore, as our main focus is on the establishment of a data-focused approach to optimizing LSTMs, we do not discuss in detail the theoretical or mathematical principles of LSTMs, and expect the reader to already be familiar with these topics. Many of the works cited in the following section provide such background knowledge. We have held our commentary on the findings of this meta-study until the Discussion section.

*Experimental Setups*

Across the 27 studies, each used a unique implementation of LSTMs for the research conducted therein. Many reports used the open-source OPPORTUNITY Activity Recognition dataset :cite:`Roggenetal2010` :cite:`OrdonezRoggen2016` :cite:`Riveraetal2017` :cite:`Gaoetal2016` :cite:`Zhaoetal2017` :cite:`Broome2017` :cite:`GuanPlotz2017`, while other datasets used include PAMAP2 :cite:`OrdonezRoggen2016` :cite:`Setterquist2018` :cite:`GuanPlotz2017` :cite:`Zhangetal2018`, Skoda :cite:`OrdonezRoggen2016` :cite:`GuanPlotz2017`, WISDM :cite:`Chenetal2016` :cite:`U2018`, ChaLearn LAP large-scale Isolated Gesture dataset (IsoGD) :cite:`Zhangetal2017`, Sheffield Kinect Gesture (SKIG) dataset :cite:`Zhangetal2017`, UCI HAR dataset :cite:`U2018` :cite:`Zhaoetal2017`, a multitude of fall-related datasets :cite:`Muscietal2018`, and various study-specific internally-collected datasets. Most studies used the Python programming language. Programming packages employed include Theano Lasagne, RNNLib, and Keras with TensorFlow. While most of the studies we examined trained models on tasks under the broad umbrella of “Activities of Daily Life” (ADL) – e.g., opening a drawer, climbing stairs, walking, or sitting down – several of the studies focused on more specific human activities such as smoking :cite:`Bergelin2017`, cross-country skiing :cite:`Rassemetal2017`, eating :cite:`Kyritsisetal2017`, nighttime scratching :cite:`Moreauetal2016`, and driving :cite:`Carvalhoetal2017`.

Numerous experimental data analysis pipelines were used, including cross validation :cite:`Lefebvreetal2015`, repeating experiments :cite:`ShinSung2016`, and various train-validation-test splitting procedures :cite:`Sjostrum2017` :cite:`WuAdu2017` :cite:`Huetal2018`.

*Preprocessing*

Before training the proposed models, each study performed some degree of preprocessing. Some reports kept preprocessing to a minimum, e.g., linear interpolation to fill missing values :cite:`OrdonezRoggen2016`, per-channel normalization :cite:`OrdonezRoggen2016` :cite:`Huetal2018`, and simple standardization :cite:`Chenetal2016`, :cite:`Zhaoetal2017`. Typically, data is standardized to have zero mean, i.e., centering the amplitude around zero :cite:`Broome2017`, and unit standard deviation, whereas Zhao, et. al. standardized the data to have 0.5 standard deviation :cite:`Zhaoetal2017`.

Other noise reduction strategies employed include kernel smoothing :cite:`Gaoetal2016`, removing the gravity component :cite:`Moreauetal2016`, applying a low-pass filter :cite:`Lefebvreetal2015`, removing the initial and last 0.5 seconds :cite:`Huetal2018`. Gao, et. al. go so far as to apply Nadaraya-Watson kernel weighted average smoothing, using the Epanachnikov quadratic kernel and 40-nearest neighbor window size :cite:`Gaoetal2016`. Moreau, et. al. used the derivative of the axis-wise gravity component in order to group together segments of data from different axes, tracking a single motion across axes as the sensor rotated during a gesture :cite:`Moreauetal2016`.

Some form of data redistribution or organization was also typical. For example, Broome 2017 and Moreau, et. al. excluded the dominant Null class as a solution to class imbalance :cite:`Broome2017`, :cite:`Moreauetal`. Lee & Cho aimed to circumvent the Null-related class imbalance by first training a model to differentiate meaningful data segments from the Null class, and subsequently training a second model to predict the specific gesture class :cite:`LeeCho2013`. Moreau, et. al. used resampling to solve class imbalance Moreauetal2016.

For feeding the data into the models, the sliding window technique was commonly used, with vast discrepancy in the optimal size of the window (reported both as units of time and number of time points) and step size. Window sizes used range from 30 :cite:`Broome2017` to 100 :cite:`Zhaoetal2016` time points, and 32 :cite:`Muscietal2018`to 5000 :cite:`Zhaoetal2017` milliseconds (ms). Using a step size between windows of 50% of the window size was typical :cite:`Rassemetal2017` :cite:`Sjostrum2017` :cite:`Broome2017` :cite:`OrdonezRoggen2016`. Finally, Guan & Plotz ran an ensemble of models, each using a random sampling of a random number of frames with varying sample lengths and starting points using a wrap-around windowing method. This method is similar to the bagging scheme of random forests and was implemented to increase robustness of the model :cite:`Guan&Plotz2017`.

Once a window is generated it must be assigned a class and labeled as such. Labeling schemes used include a jumping window technique, where the class of the last data point in the window is used as the class label :cite:`OrdonezRoggen2016` or using the majority class within the window :cite:`Broome2017`.

*Architectures*

Numerous different architectural and hyperparameter choices were made among the various studies. Most studies used two LSTM layers :cite:`OrdonezRoggen2016` :cite:`Chenetal2016` :cite:`Kyritsisetal2017` :cite:`Zhangetal2017` :cite:`Riveraetal2017` :cite:`U2018` :cite:`Zhaoetal2017` :cite:`GuanPlotz2017` :cite:`Huetal2018` :cite:`Muscietal2018`, while others used a single layer :cite:`WuAdu2017` :cite:`Broome2017` :cite:`ShinSung2016` :cite:`Carvalhoetal2017` :cite:`Zhaoetal2016` :cite:`Zhangetal2018` :cite:`Seoketal2018` or three layers :cite:`Zhaoetal2016`.

Several studies designed or utilized novel LSTM architectures that went beyond the simple tuning of hyperparameters. Before we list them, note that the term “deep” in reference to neural network architectures indicates the use of multiple layers of hidden connections; for LSTMs, an architecture generally qualifies as “deep” if it has three or more hidden layers. Architectures tested include the combination of CNNs with LSTMs such as ConvLSTM :cite:`Zhangetal2017` :cite:`Gaoetal2016` and DeepConvLSTM :cite:`OrdonezRoggen2016` :cite:`Sjostrum2017` :cite:`Broome2017`; innovations related to the connections between hidden units including the bidirectional LSTM (b-LSTM) :cite:`Rassemetal2017` :cite:`Broome2017` :cite:`Moreauetal2016` :cite:`Lefebvreetal2015`, hierarchical b-LSTM :cite:`LeeCho2012`, deep residual b-LSTM :cite:`Zhaoetal2017`, and LSTM with peephole connections (p-LSTM) :cite:`Rassemetal2017`; and other nuanced architectures such as ensemble deep LSTM :cite:`GuanPlotz2017`, weighted-average spatial LSTM (WAS-LSTM) :cite:`Zhangetal2018`, deep-Q LSTM :cite:`Seoketal2018`, and similarity-based LSTM :cite:`Fiterauetal2016`. The use of densely-connected layers before or after the LSTM layers was also common. Kyritsis, et. al. added a dense layer with ReLU activation after the LSTM layers, Zhao, et. al. included a dense layer with tanh activation after the LSTMs, and Musci, et. al. used a dense layer before and after its two LSTM layers :cite:`Kyritsisetal2017` :cite:`Zhaoetal2016` :cite:`Muscietal2018`. The WAS-LSTM, deep-Q LSTM, and the similarity-based LSTM used a combination of dense and LSTM hidden layers.

Once the number of layers is determined, the number of units per LSTM layer must be set. The number of units per layer specified by various studies range from 3 :cite:`Moreauetal2016` to 512 :cite:`Setterquist2018`. Several studies used different numbers of units for different circumstances – e.g., three units per layer for unilateral movement (one arm) and four units per layer for bilateral movement (both arms) :cite:`Moreauetal2016` or 28 units per layer for the UCI HAR dataset (lower dimensionality) versus 128 units per layer for the Opportunity dataset :cite:`Zhaoetal2017`. Others used different numbers of units for different layers of the same model – e.g., 14-14-21 for a 3-layer model :cite:`Zhaoetal2016`.

Almost all of the reports used the sigmoid activation for the recurrent connections within cells and the tanh activation function for the LSTM cell outputs, as this is the activation function used the original paper :cite:`HochreiterSchmidhuber1997`. Other activation functions used for the cell outputs include ReLU :cite:`Zhaoetal2017` :cite:`Huetal2018` and sigmoid :cite:`Zhangetal2018`.

*Training*

Once a model architecture is specified, it must be trained and the weights must be updated through a back propagation technique developed specifically for recurrent neural networks known as back-propagation through time (BPTT). Weights are often initialized using specific strategies, for example random orthogonal initialization :cite:`OrdonezRoggen2016` :cite:`Sjostrum2017`, fixed random seed :cite:`Setterquist2018`, the Glorot uniform initialization :cite:`Broome2017`, random uniform initialization within [-1, 1] :cite:`Moreauetal2016`, or using a random normal distribution :cite:`Huetal2018`. Training may occur using all the input data at once, or in mini-batches of examples. Batch sizes reported range from 32 :cite:`Riveraetal2017` :cite:`Setterquist2018` to 450 :cite:`Bergelin2017`.

To calculate the amount of change needed for each training epoch, different loss functions are used. Categorical cross-entropy is the most widely used method :cite:`OrdonezRoggen2016` :cite:`Chenetal2016` :cite:`Sjostrum2017` :cite:`Kyritsisetal2017` :cite:`Setterquist2018` :cite:`Broome2017` :cite:`Huetal2018` :cite:`Zhangetal2018`, but F1 score loss :cite:`GuanPlotz2017`, mean squared error (MSE) :cite:`Carvalhoetal2017`, and mean absolute error and root MSE :cite:`Zhaoetal2016` were also used with varying degrees of success. During back propagation, various updating rules – e.g. RMSProp :cite:`OrdonezRoggen2016` :cite:`Setterquist2018` :cite:`Broome2017`, Adam :cite:`Kyritsisetal2017` :cite:`Broome2017` :cite:`Huetal2018` :cite:`Zhangetal2018`, and Adagrad :cite:`ShinSung2016` – and learning rates – 10^-7 :cite:`ShinSung2016`, 10^-4 :cite:`Sjostrum2017`, :cite:`GuanPlotz2017`, 2e-4 :cite:`Moreauetal2016`, 5e-4 :cite:`Lefebvreetal2015`, and 10^-2 :cite:`OrdonezRoggen2016` are used.

Regularization techniques are often employed to stabilize the weight update process and avoid the problem of exploding gradients (LSTMs are not susceptible to vanishing gradients :cite:`HochreiterSchmidhuber1997`. Regularization techniques employed include weight decay of 0.9 :cite:`OrdonezRoggen2016,Sjostrum2017`; update momentum of 0.9 :cite:`Moreauetal2016`, 0.2 :cite:`Lefebvreetal2015`, or the Nesterov implementation :cite:`ShinSung2016`; dropout (forgetting the output from a proportion of units, e.g., 0.5 :cite:`OrdonezRoggen2016,Sjostrum2017` or 0.7 :cite:`Zhaoetal2016`) between various layers; batch normalization :cite:`Zhaoetal2017`; or gradient clipping using the norm :cite:`Zhaoetal2017` :cite:`Huetal2018` :cite:`Zhangetal2018`. Broome 2017 chose to use the stateful configuration for its baseline LSTM :cite:`Broome2017`. In this configuration, unit memory cell weights are maintained between each training example instead of resetting them to zero after each forward pass.

Finally, models are trained for a given number of iterations, i.e., epochs. The number of epochs specified ranged from 100 :cite:`Broome2017` to 10,000 :cite:`Huetal2018`. Many studies chose to use early stopping, which stops training once performance on the validation set has slowed or halted. This prevents overfitting, which occurs when the model learns to represent irreducible error in the training data :cite:`Garethetal2017`. Various patience schemes, specifying how many epochs with no improvement above a given threshold the model should allow, were chosen.

*Performance measures*

Once the model has been trained, it is given a set of examples it has not yet seen and makes predictions on the target class that each example belongs to. Various performance measures are used to assess the performance of the model on this test set. The measures used include the F1 score - used by most :cite:`OrdonezRoggen2016` :cite:`Broome2017` :cite:`Gaoetal2016` :cite:`Zhaoetal2017` :cite:`Broome2017`, classification error :cite:`Rassemetal2017`, accuracy :cite:`Sjostrum2017` :cite:`Setterquist2018`, and ROC :cite:`Moreauetal2016` :cite:`Huetal2018`. The use of different performance measures makes comparisons across studies difficult.

As this meta-analysis style overview has shown, there are many different model constructions being employed for HAR tasks. The lack of clear understanding for how the LSTM layers are representing this specific data and which hyperparameter choices may be better for specific problems within the field has motivated the current study.

Experimental Setup
------------------

*Data*
Many studies use the high-dimensional data from inertial sensors, which supplement accelerometer measurements with axis-wise rotation information via gyroscopes and axis-wise changes in the surrounding magnetic field via magnetometers. However, accelerometer data is ubiquitous in this field and the decreased feature space has the benefits of illuminating the robustness of classification methods used in addition to requiring lower computational complexity, making on-line and on-device classifications more feasible. As such, this report mainly focuses on the use of triaxial accelerometer data.

The primary dataset used for our experiments is the Human Activity Recognition Using Smartphones Data Set (HAR Dataset) from Anguita, et. al. :cite:`Anguitaetal2013`. This is a publicly available dataset that can be downloaded via the University of California at Irvine (UCI) online Machine Learning Repository.

*HAR Dataset*
Classes include walking, climbing stairs, descending stairs, sitting, standing and laying down. This dataset was collected from built-in accelerometers and gyroscopes in smartphones worn on the waists of participants. The collectors of this data manually extracted over 500 features from the raw data; however, this study only utilizes the raw accelerometer data itself.

A degree of preprocessing was applied to the raw signals themselves by the data collectors. The accelerometer data was recorded at 50Hz and was preprocessed to remove noise by applying a third order low pass Butterworth filter with corner frequecy of 20Hz and a median filter. The cleaned data were then separated into body motion and gravity components via a second application of a low pass Butterworth filter with 0.3Hz cuttoff. A sliding window was applied to the data using a window size of 2.56 seconds (128 time points) and a 50% stride. The data for the total accelerometer signals and the body-movement only (gravity component removed) signals are provided separately, with the data from each axis (x, y, and z) contained in a separate file. Each axis-specific file contains the data for all 30 participants and all activity classes in 128-time point (128-column) rows. The participant number and activity label corresponding to each row were contained within separate files. Finally, the data and corresponding subject and label information were split into training (70%) and testing (30%) folders.

*Preprocessing*
Preprocessing was kept to a minimum. Before any scaling or windowing was performed, the data needed to be formatted in a useful way. First, the training and testing sets were combined into a single dataset. The windows were effectively removed from the data by grouping the windows by participant and concatenating together time points from every other window, reforming contiguous time series. We then combined each axis-specific time series to form the desired triaxial data format, where each time point consists of the accelerometer values along the x-, y-, and z-axes as a 3-dimensional array. The participant to which each record belongs is kept track of so that no single participant is included in both training and testing sets. For optimizing our model architecture, we used an 80:20 training-to-testing ratio; whereas for the testing of the optimized model, we used 5-fold cross validation. After splitting into training and testing sets, the data is standardized by first fitting the the standardization parameters (i.e., mean and standard deviation) to the training data and then using these parameters to standardize the training and testing sets separately. This prevents exposing any summary information about the testing set to the model before training, i.e., data leakage. Finally, a fixed-length sliding window was applied, the windows were shuffled to avoid localization during backpropagation, and the data was ready to feed into the LSTM neural network.

*Training*
This experiment was broken up into two different sections. The first section consisted of hyperparameter optimization. In the past, we have used randomized grid search for tuning neural network hyperparameters. However, due to the vastness of the search space, it is difficult to assess even 10% of the possible architectures in a reasonable amount of time and computing resources. Thus, for this experiment we turned to heuristic-based searches. We used a tree-structured Parzen estimator (TPE) algorithm to aid in exploring the hyperparameter search space more efficiently. TPE utilizes sequential model-based optimization (SMBO) and works by iteratively re-configuring initially uniform distributions of parameter settings into weighted distributions that reflect observed higher performances in specific areas of each setting :cite:`Bergstraetal2011`.

The ranges of hyperparameters were devised to include all ranges explored by the various reports reviewed in the above section of this paper, as well as any other well-defined range or setting used in the field. The hyperparameters tested are as follows.

.. code-block:: python

  LSTM(units={{choice(numpy.arange(2,512))}},\
        activation={{choice(['softmax', 'tanh', 'sigmoid', 'relu', 'linear'])}},\
        recurrent_activation={{choice(['tanh', 'hard_sigmoid', 'sigmoid', 'relu', 'linear'])}},\
        use_bias={{choice([True, False])}},\
        kernel_initializer={{choice(['zeros', 'ones', RandomNormal(), RandomUniform(minval=-1, maxval=1), Constant(value=0.1), 'orthogonal', 'lecun_normal', 'glorot_uniform'])}},\
        recurrent_initializer={{choice(['zeros', 'ones', RandomNormal(), RandomUniform(minval=-1, maxval=1), Constant(value=0.1), 'orthogonal', 'lecun_normal', 'glorot_uniform'])}},\
        unit_forget_bias=True,\
        kernel_regularizer={{choice([None,'l2', 'l1'])}},\
        recurrent_regularizer={{choice([None,'l2', 'l1'])}},\
        bias_regularizer={{choice([None,'l2', 'l1'])}},\
        activity_regularizer={{choice([None,'l2', 'l1'])}},\
        dropout={{uniform(0, 1)}},\
        recurrent_dropout={{uniform(0, 1)}})

  adam = keras.optimizers.Adam(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
  rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
  sgd = keras.optimizers.SGD(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)

  model.compile(optimizer={{choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'nadam', 'adam'])}},
                loss='categorical_crossentropy', metrics=['accuracy'])

  results = model.fit(X_train, y_train, epochs=1000,\
                      batch_size={{choice(numpy.arange(32, 450))}},\
                      validation_split=0.2, callbacks=[early_stop, model_saver])

Due to constraints in the Python package used for hyperparameter optimization (i.e., hyperas from hyperopt), the window size, stride length and number of layers were optimized on the highest performing combination of all other hyperparameters. Thus, for initial optimization, data was partitioned using a window size of 128 with 50% stride length and fed into a 2-layer LSTM network. Subsequently, window size, stride length and number of layers were tested using randomized grid search on the following ranges:

.. code-block:: python

  window_size = [24, 48, 64, 128, 192, 256]
  stride      = [0.25*window_size, 0.5*window_size, 0.75*window_size]
  n_layers    = [1, 2, 3, 4]

For the second portion of the experiment, the highest performing model was assessed using 5-fold cross validation, where the folds were made at the participant level so that no single participant's data ended up in the training and testing sets.

*Performance Measures*
During hyperparameter optimization, back propagation was set to minimize cross-entropy. The best model was selected using the accuracy from the test trial after each training run. During cross-validation, the F1 Score and accuracy are compiled and summed across all folds.

*Hardware and Sofware*
Hyperparameter optimization took place on ____

All models were written in the Python programming language. The LSTMs were built and run using the Keras library and TensorFlow as the backend heavy lifter. Hyperas by Hyperopt was used to optimize the network. Sci-kit learn provided the packages for cross validation and randomized grid search. Numpy and Pandas were used to read and reformat the data among various other operations.

Results
-------
During preliminary testing of a baseline model to ensure the code would run, we found that the model performed better on the raw accelerometer data compared to the data with the gravity-component removed. As such, we used the total accelerometer signal in our experiment. The hyperparameter optimization explored a search space with millions of possible parameter combinations. Running the search on a __ desktop equipped with an NVIDIA GeForce GTX 1080 Graphics card, a solution was found in __ days/hours. Test accuracies ranged from 16% to __(low 90s so far). The parameters of the highest-performing model are summarized below.

We ran 5-fold CV on the optimized model and computed the overall and class-wise F1 scores and accuracies. (HMP DATA AS WELL?) The results are summarized in the following table:

Discussion
----------

*Review of previous works*
Of note, several studies split data from the same participant between training and testing data (e.g., :cite:`Huetal2018`), which we have pinpointed as a potential source of data leakage, or at least an approach less emulative of real-world situations in which the “testing” data will consist of data from newly encountered individuals.
:cite:`Zhaoetal2017`citing Wiesler, et. al. as supporting this nuance for deep learning implementations :cite:`Wiesleretal2014`. Standardization is often important for data-dependent models such as LSTM RNNs since the presence of outliers and skewed distributions may distort the weight embeddings :cite:`Garethetal2017`. Furthermore, if the common sliding window technique is used (discussed further below), standardization can be utilized for online activity classification. For these reasons, we standardize the inputs to our models in this experiment.
While these methods are not exceedingly technical or difficult to implement, they do require a degree of domain knowledge in signal processing, and are more computationally expensive and less realistic for online and on-device implementations than is desired. Much of the appeal of non-linear models such as neural networks is their ability to learn from raw data itself and independently perform smoothing and feature extraction on noisy data through parameterized embedding of the data; thus, we aim to keep preprocessing to a minimum in our experiments and instead rely on the models themselves.
Broome 2017 and Moreau, et. al. excluded the dominant Null class as a solution to class imbalance; however, this is not very feasible for real-world online activity classification, where long periods of non-activity between meaningful segments are to be expected
At this point we reiterate that we saw no explicit evidence of efforts to prevent data leakage during preprocessing. Data leakage occurs when any smoothing, grouping, filtering, or other operations are performed on the entire dataset before separating the test set. To ensure generalizability of results, if any preprocessing is to be performed on the test set, only parameters from the training set can be used. For example, when standardizing the testing set, the researcher should first separately standardize the training set and then use the mean and standard deviation of the training set as parameters for standardizing the test set. If possible, test set data should come from different participants or even different datasets than those used for the training data :cite:`Hastieetal2017`.
From our own experiments, more than three layers is not practical due to largely increased training time and overfitting of the training data.
Reimers & Gurevych emphasize the importance of weight initialization for model performance in their survey of the importance of hyperparameter tuning for using LSTMs with sequence labeling :cite:`ReimersGurevych2017`. This finding in a domain similar but not equal to HAR using accelerometer data highlights the importance of thorough comparisons of various architectures in this domain.

*Hyperparameter optimization and data analysis pipeline*
We structured our experiments

We found that results using the total accelerometer signal exceeded those obtained using the body-movement only signal with gravity component removed. This demonstrates a promising potential of non-linear data-dependent models such as neural networks to classify complex noisy data in real-time settings. Additionally, we demonstrate the ability of these models to perform competitively with benchmark experiments even after extreme care is taken to prevent any exposure of information about the testing data to the model during training or before testing.

We used two datasets for our experiments, both of which are publicly available . The datasets we used include the Public Dataset of Accelerometer Data for Human Motion Primitives Detection (HMP Dataset) from Bruno, et. al. :cite:`Brunoetal2012` and the .

*HMP Dataset*
The classes of activities included in the HMP dataset include brushing teeth, combing hair, climbing stairs, descending stairs, pouring out a glass of water, drinking from a glass, eating meat, getting out of a bed, laying down onto a bed, sitting down into a chair, standing up from a chair, using a telephone, and walking. The data consists of triaxial accelerometer data recorded at 32Hz from a device worn on the wrist. The study includes data from 16 volunteers and exhibits imbalanced proportions between classes. The data is provided on the UCI MLR website as such: each activity class has its own folder, within each of which is an individual folder for each trial recorded for that activity. Inside each trial folder is triaxial accelerometer data, axes represented as columns, scaled from 0 to 63.


Conclusion/Future Work
----------------------
HAR experiments first took place in the biomedical clinic in the late 1990’s and early 2000’s :cite:`Bussmanetal2001`, :cite:`Ravietal2005`, :cite:`Tulenetal97`, :cite:`Bussmanetal98`, and quickly spread, leading to HAR research in current-day innovative settings such as the automobile :cite:`Carvalhoetal2017`, the bedroom :cite:`Moreauetal2016`, the dining room :cite:`Kyritsisetal2017`, and outdoor sporting environments :cite:`Rassemetal2017`.
Improvements in and increased accessibility to specialized wearable sensors and sensor-equipped handheld devices, e.g., smartphones, have led to a large number of publicly available datasets and a multitude of experiments and applications using this data.
As the uses and applications of wearable accelerometry devices for human activity recognition (HAR) have expanded, so too have the corresponding machine learning methods evolved in efforts to improve performance for these tasks.
