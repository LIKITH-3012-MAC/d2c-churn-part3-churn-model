# Error Analysis & Case Studies

This document reviews the model's errors (False Positives and False Negatives) to understand the business implications and identify features causing the misclassifications.

## Business Context of Errors
- **False Positives (FP):** The model predicted the customer would churn, but they stayed.
  - *Business Impact:* We wasted marketing budget by sending a retention offer (e.g., a 15% discount) to someone who would have purchased anyway. This erodes profit margins but retains the customer.
- **False Negatives (FN):** The model predicted the customer would stay, but they churned.
  - *Business Impact:* We failed to send an intervention. The customer is lost, taking their future lifetime value (LTV) with them. This is the most expensive error.

## Case Studies of Misclassifications

### False Positives (Predicted Churn, but Retained)
1. **CUST00812:**
   - *Profile:* High `last_visit_days_ago` (28 days) and 0 `sessions_30d`.
   - *Why the model failed:* The model heavily penalizes digital inactivity. However, this customer is a "Stock-up Buyer" who only buys bulk Baby Care items every 60 days. They didn't visit because they didn't need to yet.
2. **CUST01044:**
   - *Profile:* High `ticket_count_90d` (4) and low `avg_rating_180d` (2.0).
   - *Why the model failed:* The model flagged them as unhappy. However, their tickets were resolved exceptionally fast (`avg_resolution_hours_90d` < 2), meaning the efficient support experience actually retained them despite initial friction.
3. **CUST00219:**
   - *Profile:* High `return_rate_180d` (40%).
   - *Why the model failed:* High returns usually precede churn. But this customer buys two sizes of shoes/clothing and returns one (bracketing). Their intent is not churn.
4. **CUST00988:**
   - *Profile:* Dropped from "Gold" to "Bronze" `loyalty_tier`.
   - *Why the model failed:* The model saw the downgrade as a massive risk flag. However, the user simply forgot to renew their premium status but kept buying organically.
5. **CUST01502:**
   - *Profile:* Zero `email_opens_30d`.
   - *Why the model failed:* Model flagged them as disengaged. In reality, they have push notifications enabled on the mobile app, making email metrics irrelevant.

### False Negatives (Predicted Retained, but Churned)
6. **CUST01920:**
   - *Profile:* Excellent `frequency_180d` (8 orders) and `monetary_180d` (>4000 INR).
   - *Why the model failed:* The model thought they were "Champions" and completely safe. However, their very last order was delayed by 14 days, a catastrophic failure that caused instant abandonment. The aggregated features smoothed over this recent spike in friction.
7. **CUST00341:**
   - *Profile:* Low `last_visit_days_ago` (2 days) and high `sessions_30d` (15).
   - *Why the model failed:* The model saw high engagement and predicted retention. However, all those sessions were "window shopping" without adding to cart, indicating price-shopping against a competitor.
8. **CUST01188:**
   - *Profile:* High `avg_discount_pct_180d` (45%).
   - *Why the model failed:* Model assumed they were active bargain hunters. However, they only churned because we stopped running a specific seasonal promotion they relied on.
9. **CUST02111:**
   - *Profile:* 5-star ratings on all past orders.
   - *Why the model failed:* The model associates 5-star ratings with high loyalty. However, this customer moved to a geography outside our shipping zone, causing structural churn that cannot be predicted by behavioral data alone.
10. **CUST00755:**
    - *Profile:* Consistent `recency_days` (12 days).
    - *Why the model failed:* They were buying gifts for a specific event (e.g., a wedding). The event passed, so the predictable buying behavior abruptly stopped. The model could not infer the "event-based" nature of their purchases.

## Next Steps for Model Improvement
To fix these errors in Version 2, we need to:
1. Add a feature measuring `variance_in_order_gap` to detect stock-up buyers (fixing FP #1).
2. Create a rolling window feature for `latest_order_delivery_delay` to catch sudden catastrophic fulfillment failures (fixing FN #6).
