# This repo corresponds to [my Medium post!](https://medium.com/@domvandendries/using-ml-to-predict-lifetimes-of-new-customers-685051ff75f8?source=friends_link&sk=79600802708579d86df13eee69aceb34)


## Intro & Motivation
When analyzing processes like churn and customer lifecycles, it’s so tempting to just export your existing customer base, whip up a few features like *average_order_value*, *order_frequency* and *n_total_orders*, throw it in a model and call it a day. Your model probably performs pretty good too! But using ‘present-tense’ data to predict a ‘past-tense’ outcome doesn’t make much sense. Plus, it stinks of __survivorship bias__ and __data leakage__!

What if you were able to estimate a brand-new customer’s lifetime, just by looking at their first order? Is this even possible with such little data? This is known as the “cold start problem” — to solve it, we’ll need to figure out a way to best leverage our severely limited data. This tutorial will walk you through estimating the lifetimes of new customers using a Gradient Boost Regressor.

Shameless plug: click [here](http://52.90.122.192:1212/churning_man) to check out the interactive webapp version of my model — enjoy the 90's HTML and CSS.

# Intrigued? Read the rest on [Medium!](https://medium.com/@domvandendries/using-ml-to-predict-lifetimes-of-new-customers-685051ff75f8?source=friends_link&sk=79600802708579d86df13eee69aceb34)
