# product-matching
[Shopee NSDC Product Matching 2020](https://www.kaggle.com/c/pre-product-matching-id-ndsc-2020/)
At Shopee, we strive to ensure fairness to both buyers and sellers, and improve user experience by providing high quality products with low price. Lowest Price Guaranteed feature is one of our assurances that some of the items sold in shopee is the cheapest among competitors. The fundamental data science technique for this feature is product matching, where we are applying a machine learning model to automatically detect if two items, based on their product information such as title, description, and images, are actually the same product. Besides low price guarantee, product matching also benefits buyers and sellers in other aspects. From a buyer's point of view, when a buyer searches for a product using keywords, retrieved listings of same or similar products could be shown in different strategies (e.g. items of lower price on top or items of best match on top). From a seller’s point of view, if the seller uploads a new item, we could recommend under which category this item could be placed, based on it’s similar or same products’ category information. If the same products are put under different categories, one of them could potentially cause a category spam problem. Therefore, product matching has a great impact and importance on guaranteeing Shopee’s products high quality and good user experience.

## Task
Given the item pairs, to build a model to predict if they are the same or different products.

## Requirements
```
pip install -r requirements.txt
```
