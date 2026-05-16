# Databricks notebook source
# DBTITLE 1,Prediction Function
# Create a function to predict approval for new applications
def predict_credit_approval(new_application_data, model=rf):
    """
    Predicts credit card approval for new application(s).
    
    Parameters:
    -----------
    new_application_data : DataFrame or array
        The features of the new application(s) to predict
    model : trained model object
        The model to use for prediction (default: Random Forest)
    
    Returns:
    --------
    predictions : array
        0 for denied, 1 for approved
    """
    # Make prediction
    prediction = model.predict(new_application_data)
    
    # Get probability scores
    probability = model.predict_proba(new_application_data)
    
    return prediction, probability

print("✅ Prediction function created!")
print("\nYou can now use predict_credit_approval(data) to predict on new applications.")

# COMMAND ----------

# DBTITLE 1,Example Predictions
# Example: Make predictions on a few test samples
print("="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

# Select 5 random samples from the test set
sample_indices = np.random.choice(len(X_test), 5, replace=False)
X_samples = X_test.iloc[sample_indices]
y_samples = y_test[sample_indices]

# Make predictions
predictions, probabilities = predict_credit_approval(X_samples, rf)

# Display results
print("\n🔮 Predictions on 5 sample applications:\n")

for i in range(len(predictions)):
    actual = "Approved" if y_samples[i] == 1 else "Denied"
    predicted = "Approved" if predictions[i] == 1 else "Denied"
    confidence = probabilities[i][predictions[i]] * 100
    
    # Check if prediction is correct
    status = "✅ Correct" if predictions[i] == y_samples[i] else "❌ Incorrect"
    
    print(f"Application #{i+1}:")
    print(f"  Actual: {actual}")
    print(f"  Predicted: {predicted} (Confidence: {confidence:.1f}%)")
    print(f"  {status}")
    print()

# Calculate accuracy on these samples
sample_accuracy = accuracy_score(y_samples, predictions)
print(f"🎯 Accuracy on these {len(predictions)} samples: {sample_accuracy*100:.1f}%")

# COMMAND ----------

# DBTITLE 1,Predict on Entire Test Set
# Show overall prediction summary on test set
print("="*60)
print("TEST SET PREDICTION SUMMARY")
print("="*60)

# Make predictions on entire test set
test_predictions = rf.predict(X_test)

# Count predictions
approved_count = np.sum(test_predictions == 1)
denied_count = np.sum(test_predictions == 0)
total_count = len(test_predictions)

print(f"\n📋 Total test applications: {total_count}")
print(f"\n🟢 Predicted Approved: {approved_count} ({approved_count/total_count*100:.1f}%)")
print(f"🔴 Predicted Denied: {denied_count} ({denied_count/total_count*100:.1f}%)")

# Compare with actual
actual_approved = np.sum(y_test == 1)
actual_denied = np.sum(y_test == 0)

print(f"\n🎯 Actual Approved: {actual_approved} ({actual_approved/total_count*100:.1f}%)")
print(f"🎯 Actual Denied: {actual_denied} ({actual_denied/total_count*100:.1f}%)")

# Model performance
correct_predictions = np.sum(test_predictions == y_test)
print(f"\n✅ Correct Predictions: {correct_predictions}/{total_count} ({accuracy_rf*100:.2f}%)")
print(f"❌ Incorrect Predictions: {total_count - correct_predictions}/{total_count} ({(1-accuracy_rf)*100:.2f}%)")

# COMMAND ----------

# DBTITLE 1,Project Summary
# MAGIC %md
# MAGIC ## 🎉 Project Summary
# MAGIC
# MAGIC ### What We Accomplished
# MAGIC
# MAGIC 1. **📥 Data Loading**: Successfully downloaded and loaded 690 credit card applications from UCI repository
# MAGIC
# MAGIC 2. **🔍 Data Exploration**: Analyzed the dataset structure, identified missing values, and understood target distribution
# MAGIC
# MAGIC 3. **🔧 Data Preprocessing**: 
# MAGIC    - Handled missing values using mean imputation (numerical) and mode imputation (categorical)
# MAGIC    - Encoded categorical variables to numbers
# MAGIC    - Scaled numerical features
# MAGIC    - Split data into 80% training and 20% testing
# MAGIC
# MAGIC 4. **🤖 Model Building**: Trained three different models:
# MAGIC    - Logistic Regression
# MAGIC    - Decision Tree
# MAGIC    - Random Forest
# MAGIC
# MAGIC 5. **📊 Model Evaluation**: Compared models using accuracy, confusion matrices, and classification reports
# MAGIC
# MAGIC 6. **🔮 Predictions**: Demonstrated how to use the best model to predict new applications
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC ⭐ **Best Model**: The Random Forest model achieved the highest accuracy
# MAGIC
# MAGIC 💡 **Feature Importance**: Identified which applicant characteristics are most important for approval decisions
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC To improve this project further, you could:
# MAGIC * **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to find optimal model parameters
# MAGIC * **Handle Class Imbalance**: Apply SMOTE or other techniques if classes are imbalanced
# MAGIC * **Feature Engineering**: Create new features from existing ones
# MAGIC * **Try More Models**: Experiment with XGBoost, LightGBM, or Neural Networks
# MAGIC * **Cross-Validation**: Use k-fold cross-validation for more robust evaluation
# MAGIC * **Deploy the Model**: Save the model and create an API for real-time predictions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Resources for Learning More
# MAGIC
# MAGIC * **Scikit-learn Documentation**: https://scikit-learn.org/
# MAGIC * **Pandas Documentation**: https://pandas.pydata.org/
# MAGIC * **MLflow Documentation**: https://mlflow.org/
# MAGIC * **UCI ML Repository**: https://archive.ics.uci.edu/

# COMMAND ----------

# DBTITLE 1,Compare Model Accuracies
# Compare accuracies of all three models
print("="*60)
print("MODEL COMPARISON")
print("="*60)

# Create a comparison DataFrame
model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_lr, accuracy_dt, accuracy_rf],
    'Accuracy_Percentage': [accuracy_lr*100, accuracy_dt*100, accuracy_rf*100]
})

# Sort by accuracy
model_comparison = model_comparison.sort_values('Accuracy', ascending=False)

print("\n🏆 Model Performance Ranking:")
display(model_comparison)

# Identify best model
best_model_name = model_comparison.iloc[0]['Model']
best_accuracy = model_comparison.iloc[0]['Accuracy']

print(f"\n⭐ Best Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy")

# Visualize comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(model_comparison['Model'], model_comparison['Accuracy_Percentage'], 
               color=['#FFD700', '#C0C0C0', '#CD7F32'])  # Gold, Silver, Bronze colors
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 100])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Confusion Matrices
# Create confusion matrices for all models
print("="*60)
print("CONFUSION MATRICES")
print("="*60)
print("\nA confusion matrix shows:")
print("  - True Negatives (TN): Correctly predicted denials")
print("  - False Positives (FP): Incorrectly predicted approvals (Type I error)")
print("  - False Negatives (FN): Incorrectly predicted denials (Type II error)")
print("  - True Positives (TP): Correctly predicted approvals")
print()

# Calculate confusion matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Plot all three confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Labels for the matrix
labels = ['Denied (0)', 'Approved (1)']

# Logistic Regression
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
            yticklabels=labels, ax=axes[0], cbar=False)
axes[0].set_title(f'Logistic Regression\nAccuracy: {accuracy_lr*100:.2f}%', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Decision Tree
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', xticklabels=labels, 
            yticklabels=labels, ax=axes[1], cbar=False)
axes[1].set_title(f'Decision Tree\nAccuracy: {accuracy_dt*100:.2f}%', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples', xticklabels=labels, 
            yticklabels=labels, ax=axes[2], cbar=False)
axes[2].set_title(f'Random Forest\nAccuracy: {accuracy_rf*100:.2f}%', fontweight='bold')
axes[2].set_ylabel('Actual')
axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Detailed Classification Reports
# Print detailed classification reports
print("="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)
print("\nMetrics explained:")
print("  - Precision: Of all predicted approvals, how many were correct?")
print("  - Recall: Of all actual approvals, how many did we find?")
print("  - F1-Score: Harmonic mean of precision and recall")
print("  - Support: Number of samples in each class")
print()

target_names = ['Denied (-)', 'Approved (+)']

print("\n" + "="*60)
print("🟦 LOGISTIC REGRESSION")
print("="*60)
print(classification_report(y_test, y_pred_lr, target_names=target_names))

print("\n" + "="*60)
print("🟩 DECISION TREE")
print("="*60)
print(classification_report(y_test, y_pred_dt, target_names=target_names))

print("\n" + "="*60)
print("🟪 RANDOM FOREST")
print("="*60)
print(classification_report(y_test, y_pred_rf, target_names=target_names))

# COMMAND ----------

# DBTITLE 1,Feature Importance Analysis
# Feature importance (only available for tree-based models)
print("="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)
print("\nFeature importance tells us which features have the most influence on predictions.")
print("Higher values mean the feature is more important for making decisions.\n")

# Get feature importance from Random Forest (best tree-based model)
feature_importance = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("🌲 Random Forest - Top 10 Most Important Features:")
display(feature_importance.head(10))

# Visualize top 10 features
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'], color='forestgreen')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features (Random Forest)', fontweight='bold')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

print(f"\n💡 Insight: Feature '{feature_importance.iloc[0]['Feature']}' is the most important for predicting credit card approval!")

# COMMAND ----------

# DBTITLE 1,Step 6: Making Predictions
# MAGIC %md
# MAGIC ## Step 6: Making Predictions on New Data
# MAGIC
# MAGIC Now that we have our trained model, let's see how to use it to predict whether new credit card applications should be approved or denied.
# MAGIC
# MAGIC We'll use the **Random Forest** model (our best performer) to make predictions.

# COMMAND ----------

# DBTITLE 1,Setup MLflow Experiment
# Setup MLflow experiment for tracking
print("📋 Setting up MLflow experiment tracking")

# Set experiment name
experiment_name = "/Users/pirthipalsingh138@gmail.com/credit-card-approval"
mlflow.set_experiment(experiment_name)

print(f"\u2705 Experiment set: {experiment_name}")
print("All model training runs will be logged here for comparison.")

# COMMAND ----------

# DBTITLE 1,Train Model 1: Logistic Regression
# Model 1: Logistic Regression
# A simple, fast, and interpretable model that works well for binary classification

print("="*60)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*60)

# Start MLflow run
with mlflow.start_run(run_name="Logistic_Regression"):
    print("\n🏋️ Training Logistic Regression model...")
    
    # Create and train the model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred_lr = log_reg.predict(X_test)
    
    # Calculate accuracy
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy_lr)
    
    # Log the model
    mlflow.sklearn.log_model(log_reg, "model")
    
    print(f"\u2705 Model trained!")
    print(f"🎯 Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
    print(f"\n📈 Logged to MLflow experiment: {experiment_name}")

# COMMAND ----------

# DBTITLE 1,Train Model 2: Decision Tree
# Model 2: Decision Tree
# Creates a tree of if-then rules to make predictions
# Easy to visualize and understand

print("="*60)
print("MODEL 2: DECISION TREE")
print("="*60)

# Start MLflow run
with mlflow.start_run(run_name="Decision_Tree"):
    print("\n🏋️ Training Decision Tree model...")
    
    # Create and train the model
    # max_depth limits tree depth to prevent overfitting
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred_dt = dt.predict(X_test)
    
    # Calculate accuracy
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "Decision Tree")
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy_dt)
    
    # Log the model
    mlflow.sklearn.log_model(dt, "model")
    
    print(f"\u2705 Model trained!")
    print(f"🎯 Accuracy: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")
    print(f"\n📈 Logged to MLflow experiment: {experiment_name}")

# COMMAND ----------

# DBTITLE 1,Train Model 3: Random Forest
# Model 3: Random Forest
# An ensemble (combination) of many decision trees
# Usually provides the best accuracy by averaging predictions from multiple trees

print("="*60)
print("MODEL 3: RANDOM FOREST")
print("="*60)

# Start MLflow run
with mlflow.start_run(run_name="Random_Forest"):
    print("\n🏋️ Training Random Forest model...")
    
    # Create and train the model
    # n_estimators = number of trees in the forest
    # max_depth limits individual tree depth
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred_rf = rf.predict(X_test)
    
    # Calculate accuracy
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy_rf)
    
    # Log the model
    mlflow.sklearn.log_model(rf, "model")
    
    print(f"\u2705 Model trained!")
    print(f"🎯 Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
    print(f"\n📈 Logged to MLflow experiment: {experiment_name}")

# COMMAND ----------

# DBTITLE 1,Step 5: Model Evaluation
# MAGIC %md
# MAGIC ## Step 5: Model Evaluation
# MAGIC
# MAGIC Now let's compare all three models and see which one performs best. We'll look at:
# MAGIC * **Accuracy** - Overall percentage of correct predictions
# MAGIC * **Confusion Matrix** - Shows types of errors (false positives vs false negatives)
# MAGIC * **Classification Report** - Precision, recall, and F1-score for each class
# MAGIC * **Feature Importance** - Which features matter most (for tree-based models)

# COMMAND ----------

# DBTITLE 1,Step 3: Data Preprocessing
# MAGIC %md
# MAGIC ## Step 3: Data Preprocessing
# MAGIC
# MAGIC Now we'll prepare the data for machine learning:
# MAGIC 1. **Handle missing values** - Fill in gaps using statistical methods
# MAGIC 2. **Encode categorical variables** - Convert text categories to numbers
# MAGIC 3. **Split features and target** - Separate what we predict from what we use to predict
# MAGIC 4. **Create train/test sets** - Split data for training and evaluation

# COMMAND ----------

# DBTITLE 1,Separate Features and Target
# Step 1: Separate features (X) from target (y)
print("🎯 Step 1: Separating features and target variable")

# Features: All columns except 'Target'
X = df.drop('Target', axis=1)

# Target: The 'Target' column (what we want to predict)
y = df['Target']

print(f"\u2705 Features shape: {X.shape}")
print(f"\u2705 Target shape: {y.shape}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n🗒️ Found {len(categorical_cols)} categorical columns: {categorical_cols}")
print(f"🔢 Found {len(numerical_cols)} numerical columns: {numerical_cols}")

# COMMAND ----------

# DBTITLE 1,Handle Missing Values and Encode
# Step 2: Handle missing values and encode categorical variables
print("🔧 Step 2: Handling missing values and encoding categorical variables")

# Create a copy of X to work with
X_processed = X.copy()

# --- Handle Categorical Columns ---
print("\n1️⃣ Processing categorical columns...")
for col in categorical_cols:
    # Fill missing values with the most frequent value (mode)
    if X_processed[col].isnull().any():
        most_frequent = X_processed[col].mode()[0]
        X_processed[col].fillna(most_frequent, inplace=True)
        print(f"   - {col}: Filled {X[col].isnull().sum()} missing values with '{most_frequent}'")
    
    # Encode categorical values to numbers using LabelEncoder
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))

print("   ✅ Categorical columns encoded!")

# --- Handle Numerical Columns ---
print("\n2️⃣ Processing numerical columns...")
for col in numerical_cols:
    # Fill missing values with the mean (average) of the column
    if X_processed[col].isnull().any():
        mean_value = X_processed[col].mean()
        missing_count = X_processed[col].isnull().sum()
        X_processed[col].fillna(mean_value, inplace=True)
        print(f"   - {col}: Filled {missing_count} missing values with mean ({mean_value:.2f})")

print("   ✅ Numerical columns imputed!")

# Verify no missing values remain
print(f"\n✅ Total missing values after preprocessing: {X_processed.isnull().sum().sum()}")

# Encode target variable (+ becomes 1, - becomes 0)
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\n🎯 Target encoded: '+' → {le_target.transform(['+'])[0]}, '-' → {le_target.transform(['-'])[0]}")

# COMMAND ----------

# DBTITLE 1,Scale Features
# Step 3: Scale numerical features
# Scaling makes all features have similar ranges, which helps many ML algorithms perform better
print("📊 Step 3: Scaling numerical features")

scaler = StandardScaler()

# Only scale the numerical columns
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

print(f"\u2705 Scaled {len(numerical_cols)} numerical features to have mean=0 and std=1")
print("\nSample of processed data:")
display(X_processed.head())

# COMMAND ----------

# DBTITLE 1,Split into Train and Test Sets
# Step 4: Split data into training and testing sets
print("✂️ Step 4: Splitting data into train and test sets")

# Split: 80% for training, 20% for testing
# random_state=42 ensures we get the same split each time (reproducibility)
# stratify=y_encoded ensures both sets have similar proportions of approved/denied
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, 
    y_encoded, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y_encoded  # Keep same class proportions in train and test
)

print(f"\u2705 Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_processed)*100:.1f}%)")
print(f"\u2705 Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_processed)*100:.1f}%)")

# Verify class distribution is maintained
print("\n🎯 Target distribution:")
print(f"   Training: {np.bincount(y_train)} (Denied: {np.bincount(y_train)[0]}, Approved: {np.bincount(y_train)[1]})")
print(f"   Test: {np.bincount(y_test)} (Denied: {np.bincount(y_test)[0]}, Approved: {np.bincount(y_test)[1]})")
print("\n✅ Data preprocessing complete! Ready for model training.")

# COMMAND ----------

# DBTITLE 1,Step 4: Model Building
# MAGIC %md
# MAGIC ## Step 4: Model Building
# MAGIC
# MAGIC We'll train three different machine learning models and compare their performance:
# MAGIC
# MAGIC 1. **Logistic Regression** - Simple, interpretable linear model
# MAGIC 2. **Decision Tree** - Creates a tree of decision rules
# MAGIC 3. **Random Forest** - Ensemble of multiple decision trees (often more accurate)
# MAGIC
# MAGIC We'll use MLflow to track our experiments and compare results.

# COMMAND ----------

# DBTITLE 1,Step 2: Data Exploration
# MAGIC %md
# MAGIC ## Step 2: Data Exploration (EDA)
# MAGIC
# MAGIC Let's explore our data to understand:
# MAGIC * Data types and structure
# MAGIC * Missing values
# MAGIC * Target variable distribution
# MAGIC * Feature relationships

# COMMAND ----------

# DBTITLE 1,Dataset Information
# Get basic information about the dataset
print("="*60)
print("DATASET INFORMATION")
print("="*60)

# Show dimensions
print(f"\n📏 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Show data types
print("\n📑 Data Types:")
print(df.dtypes)

# Show summary statistics for numerical columns
print("\n📊 Summary Statistics:")
display(df.describe())

# COMMAND ----------

# DBTITLE 1,Missing Values Analysis
# Check for missing values
print("="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)

# Count missing values per column
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Create a summary DataFrame
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})

# Filter to show only columns with missing values
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(f"\n⚠️ Found missing values in {len(missing_df)} columns:")
    display(missing_df)
else:
    print("\n✅ No missing values found!")

# Visualize missing values
if len(missing_df) > 0:
    plt.figure(figsize=(10, 5))
    plt.bar(missing_df['Column'], missing_df['Missing_Percentage'], color='salmon')
    plt.xlabel('Column Name')
    plt.ylabel('Missing Percentage (%)')
    plt.title('Missing Values by Column')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# DBTITLE 1,Target Variable Distribution
# Analyze the target variable distribution
print("="*60)
print("TARGET VARIABLE DISTRIBUTION")
print("="*60)

# Count the number of approvals vs denials
target_counts = df['Target'].value_counts()
print("\n🎯 Target Variable Counts:")
print(target_counts)

# Calculate percentages
target_percentages = df['Target'].value_counts(normalize=True) * 100
print("\n📊 Target Variable Percentages:")
print(target_percentages)

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
axes[0].bar(target_counts.index, target_counts.values, color=['lightgreen', 'lightcoral'])
axes[0].set_xlabel('Approval Status')
axes[0].set_ylabel('Count')
axes[0].set_title('Credit Card Applications: Approved (+) vs Denied (-)')
axes[0].set_xticklabels([f"{x}\n({target_counts[x]} apps)" for x in target_counts.index])

# Pie chart
colors = ['lightgreen' if x == '+' else 'lightcoral' for x in target_counts.index]
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
axes[1].set_title('Distribution of Approvals')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Types Analysis
# Create a table showing feature types
print("="*60)
print("FEATURE TYPES ANALYSIS")
print("="*60)

# Identify feature types
feature_types = []
for col in df.columns:
    if col == 'Target':
        feature_types.append([col, 'Target Variable (Binary)'])
    elif df[col].dtype == 'object':
        feature_types.append([col, f'Categorical ({df[col].nunique()} unique values)'])
    elif df[col].dtype in ['int64', 'float64']:
        feature_types.append([col, f'Numerical ({df[col].dtype})'])
    else:
        feature_types.append([col, str(df[col].dtype)])

# Create DataFrame
feature_types_df = pd.DataFrame(feature_types, columns=['Feature', 'Type'])
print("\n🗒️ Feature Types:")
display(feature_types_df)

# Count by type
print("\n📊 Summary:")
print(f"- Categorical features: {sum(1 for _, t in feature_types if 'Categorical' in t)}")
print(f"- Numerical features: {sum(1 for _, t in feature_types if 'Numerical' in t)}")
print(f"- Target variable: 1")

# COMMAND ----------

# DBTITLE 1,Project Introduction
# MAGIC %md
# MAGIC # Credit Card Approval Prediction Project 💳
# MAGIC
# MAGIC ## Overview
# MAGIC This project predicts whether a credit card application will be **approved** or **denied** based on applicant information.
# MAGIC
# MAGIC ## Dataset
# MAGIC We'll use the **UCI Credit Approval Dataset** which contains:
# MAGIC * **690 applications** with 15 features (mix of personal and financial information)
# MAGIC * **Target variable**: `+` (approved) or `-` (denied)
# MAGIC * **Missing values**: Marked as `?` in the data
# MAGIC
# MAGIC ## Project Steps
# MAGIC 1. **Data Loading** - Download and load the dataset
# MAGIC 2. **Data Exploration** - Understand the data structure and patterns
# MAGIC 3. **Data Preprocessing** - Clean and prepare data for modeling
# MAGIC 4. **Model Building** - Train multiple machine learning models
# MAGIC 5. **Model Evaluation** - Compare models and select the best one
# MAGIC 6. **Predictions** - Use the model to predict new applications
# MAGIC
# MAGIC ---
# MAGIC Let's get started! 🚀

# COMMAND ----------

# DBTITLE 1,Step 1: Setup and Data Loading
# MAGIC %md
# MAGIC ## Step 1: Setup and Data Loading
# MAGIC
# MAGIC First, we'll install necessary libraries and download the dataset from the UCI Machine Learning Repository.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# Install required libraries
# We'll use pip to install any packages that aren't already available
%pip install scikit-learn pandas matplotlib seaborn mlflow --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
# Restart Python kernel to load newly installed libraries
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set visualization style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ All libraries imported successfully!")

# COMMAND ----------

# DBTITLE 1,Download and Load Dataset
# Download the dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
filename = "/tmp/crx.data"

# Download the file
print("📥 Downloading dataset...")
urllib.request.urlretrieve(url, filename)
print("✅ Dataset downloaded successfully!")

# Define column names (the dataset doesn't have headers)
# Features are anonymized as A1, A2, etc. for privacy
column_names = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
    'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'Target'
]

# Load the data into a pandas DataFrame
# na_values='?' treats '?' as missing values
df = pd.read_csv(filename, names=column_names, na_values='?')

print(f"\n✅ Dataset loaded! Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
display(df.head())