## Price performance prediction of single stocks using fundamental data ##

Fundamental data includes values financial ratios like PE, revenues, market capitals etc.
Time series regression is performed using various machine learning methods on a set of technology stocks. 

A baseline performance of using the pervious data as prediction is used to compare with that of Random Forest, Light LGB

### What is this repository for? ###

Deep learning for various basic examples and ultimately for fundamental analysis in stock market.

### How do I get set up? ###

Reference 
https://github.com/bzamecnik/deep-instrument-heroku
https://realpython.com/blog/python/flask-by-example-part-1-project-setup/

Add git remote:

e.g. heroku git:remote -a deepfundamental-stage

Fix gunicorn not found:
heroku run pip install gunicorn

Deploy to Heroku(with piplines dev/staging):
1. heroku create deepfundamental-staging --remote dev
2. heroku fork -a deepfundamental-staging deepfundamental
3. git remote add staging https://git.heroku.com/deepfundamental-staging.git
4. git push staging master
5. (create deepfundamental similarly)
6. heroku pipelines:create -a deepfundamental
7. heroku pipelines:add -a deepfundamental-staging deepfundamental
8. heroku pipelines:add -a deepfundamental deepfundamental
9. heroku pipelines:promote -r staging

### Contribution guidelines ###


