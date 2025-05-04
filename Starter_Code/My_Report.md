Justin Zisholtz

Overview of the Analysis

The purpose of this analysis was to build and evaluate a binary classification model for Alphabet Soup, a nonprofit foundation. The goal was to help the organization predict which applicants for funding are most likely to be successful in their ventures. We used a neural network model built with TensorFlow and Keras, trained on historical data from over 34,000 funded organizations.

Results:

Data Preprocessing
•	Target variable:
o	    IS_SUCCESSFUL — this binary column indicates whether the organization used the funding successfully.
•	Feature variables:
o	    All columns except EIN, NAME, and IS_SUCCESSFUL. These include:
            - APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
•	Removed variables:
o	    EIN and NAME — identifiers that did not contribute to the model's learning.
•	Preprocessing steps performed:
    o	Encoded categorical variables using pd.get_dummies().
    o	Replaced rare categorical values with "Other" if they occurred fewer than a specific threshold (less than 500 instances).
    o	Split data into training and testing sets using train_test_split.
    o	Scaled the numerical features using StandardScaler.
Compiling, Training, and Evaluating the Model:
•	    Neural network architecture:
    o	Input Layer: Number of features (based on the one-hot encoded columns).
    o	Hidden Layer 1: 80 neurons, ReLU activation
    o	Hidden Layer 2: 30 neurons, ReLU activation
    o	Output Layer: 1 neuron, Sigmoid activation (binary classification)
•	Model performance:
    o	Initial Model Accuracy: ~72.5%
    o	Loss: ~0.55
•	Optimization attempts:
    1.	Increased hidden layer neurons: Tried 100 and 50 neurons in two layers.
    2.	Added a third hidden layer: Improved complexity.
    3.	Tried different activation functions (tanh, leaky ReLU): Some minor improvements.
    4.	Dropped more sparse columns like SPECIAL_CONSIDERATIONS: Simplified the input.
    5.	Tuned epochs and batch size: Increased epochs to 100 with early stopping.
•	Best performing model:
    o	Accuracy: 76.1%
    o	Saved as: AlphabetSoupCharity_Optimization.keras

Summary:
The optimized neural network model successfully reached the target of over 75% accuracy. The model shows promise as a tool for Alphabet Soup to evaluate funding applications. While neural networks provide flexibility and power for complex patterns, their interpretability is limited.

Recommendation:
To further improve or supplement this model, a tree-based model such as XGBoost or Random Forest could be used. These models often perform well on structured/tabular data and offer better interpretability (e.g., feature importance). Given the nature of the data and the business need for trust and explanation, tree-based models could complement or even outperform deep learning in this scenario.
