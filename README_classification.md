# Data Classification Using AI

A basic supervised machine learning project that builds a classification model to predict e-commerce order status using Python and scikit-learn.

---

## Goal

Predict whether an order will be Delivered, Shipped, Cancelled, Returned, or Pending — based on order attributes like product type, quantity, price, and payment method.

---

## Dataset

- **Source:** E-commerce orders dataset
- **Size:** 1,200 rows, 14 columns
- **Target column:** `OrderStatus` (5 categories)
- **Distribution:** Roughly equal across all 5 categories (~230-250 each)

### Columns Used for Prediction
| Column | Type | Description |
|---|---|---|
| Quantity | Numeric | Number of items ordered |
| UnitPrice | Numeric | Price per item |
| TotalPrice | Numeric | Total order value |
| ItemsInCart | Numeric | Items in cart at time of order |
| Product | Categorical | Type of product ordered |
| PaymentMethod | Categorical | How the customer paid |
| ReferralSource | Categorical | How the customer found the store |

### Columns Dropped
`OrderID`, `Date`, `CustomerID`, `ShippingAddress`, `TrackingNumber`, `CouponCode` — these are unique identifiers or irrelevant to prediction.

---

## How It Works

**Step 1 — Load the data**
```python
df = pd.read_csv('dataset.csv')
```

**Step 2 — Clean and encode**
- Drop irrelevant columns
- Convert categorical columns to numbers using `pd.get_dummies()`
- Result: 21 numeric features

**Step 3 — Split the data**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 80% used for training (960 rows)
- 20% used for testing (240 rows)

**Step 4 — Train the model**
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Step 5 — Evaluate**
```python
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
```

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 19% |
| Expected random baseline | 20% |

The model performs at roughly random-guessing level.

---

## Why the Accuracy Is Low

This is the most important part of the project.

- The features available (product type, payment method, referral source) have **no real causal relationship** with order status
- Whether an order gets cancelled or delivered depends on factors not in this dataset — supplier reliability, delivery zones, fraud detection, customer behavior history
- This is a **data quality problem**, not a model problem
- Switching to a different algorithm would not fix this — the signal simply isn't there

### What Would Actually Improve This

- Delivery time estimates
- Seller or warehouse location data
- Customer return history
- Stock availability at time of order

---

## Key Concepts Learned

- What supervised classification is and how it works
- How to prepare real-world messy data for ML
- What `train_test_split` does and why it matters
- How to interpret a classification report (precision, recall, f1-score)
- Why low accuracy isn't always a model failure — sometimes it's a data failure

---

## Tools Used

- Python 3
- pandas
- scikit-learn
- Jupyter Notebook

---

## How to Run

1. Clone the repository
2. Place the dataset CSV in the same folder as the notebook
3. Open `classification.ipynb` in Jupyter Notebook
4. Run all cells in order
