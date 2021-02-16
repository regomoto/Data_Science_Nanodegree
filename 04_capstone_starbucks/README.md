# Starbucks Capstone Project (Recommendation Systems)
This project is part of the Data Scientist Nanodegree from Udacity. 

## Project Overview

The purpose of this project is to determine which coupons we should offer customers to drive an increase in sales. We can do this by offering customers coupons they will most likely redeem which will increase foot traffic and revenue. 

## Problem Statement

Each of our customers are unique. They have different profiles, preferences, and purchasing habits. Because of these things, it is difficult to execute effective marketing campaigns using offers/coupons. Coupons can be an effective way to increase foot traffic in our stores, increase user engagement, and ideally increase our revenue and same store sales growth. In order to accomplish these objectives, we need to effectively personalize coupons so customers want to come into our stores and make purchases. 

## Data

We have three pieces of data avaiable to assist us in creating a prediction model:
1. Customer Profile Data
2. Coupon Data
3. Transaction History


## Proposed Solution and Metrics

This project uses FunkSVD as a recommendor system. I decided to use this since it is able to make predictions even if a user has not interacted with a coupon before. 

The metric used to evaluate the model is sum square error (SSE). I decided to use this metric to calculate the difference between predicted and actual values for the model. This allows for comparison between models and let's us see if improvements are being made for predictions. Minimizing this metric is ideal and my target is to have a model with less than 4%


# Conclusion

The final solution is a model that provides a coupon ID that a specific user will most likely interact with and end up using in the store. We can use this recommendation to push to the customer for the next release of coupon offers. 

The SSE for the final model is 3.5%, which is fairly good. 100 iterations were made to get to this number. 

Some limitations of the model include:
- If a customer really likes one coupon and continues to use it, this model may continue to offer the same coupon over and over. To mitigate this, I would make sure to store information of the last coupon a customer used. If the last coupon used is what is being recommended, then this time we can recommend the second best coupon the model suggests for the customer. 
- Does not fully take into account how coupon usage affects profitability. Each coupon eats into a sale, and ideally you want to drive customer growth to the point where customers contiuously come into the store and make purchases because they like your product, not just because they have a good coupon. This could be something discussed with management to determine how to handle this in the future and if we want to focus on the bottom line more than top line revenue growth. 

These limitations can be improved upon on a later date. 
