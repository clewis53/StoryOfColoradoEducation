StoryOfColoradoEducation
==============================

Springboard Data Science Career Track Capstone 2

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

#### Problem Statement
What is the state of education in Colorado currently and can it be improved for vulnerable students?

#### Context
Inspired by the 2013 Kaggle competition, [Visualize the State of Public Education in Colorado](https://www.kaggle.com/competitions/visualize-the-state-of-education-in-colorado/overview/description), hosted by [ColoradoSchoolGrades](https://www.facebook.com/people/Colorado-School-Grades/100080716453043/), we will be creating a mock project that would assist the state board of education to make data-driven decisions to improve student’s performance and college-readiness. They are seeking fresh perspectives, so they have made data available for outside parties to create stories from.

#### Success Criteria
In our story, we will produce a visually engaging presentation that explains how schools in the state have been performing, makes recommendations for outreach programs for students in minorities and lower-income brackets, demonstrates if expenditures have an affect on performance, and identifies patterns of performance in the statement.

#### Scope
The goal of the project is only to create a story of how performance got to where it is today and to identify possible leads for how it could be better tomorrow 

#### Constraints
We are limited to study school performance data only from the years 2010-2012. We cannot know all the measures that districts employed during these years or their effect on performance. We are not experts of education and are creating a story with limited knowledge of the features. 

#### Stakeholders
The key stakeholder here is The Colorado State Board of Education. Ultimately, other stakeholders include district boards, school boards, teachers, parents, and students.

#### Data
1. [Visualize the State of Public Education in Colorado | Kaggle](https://www.kaggle.com/competitions/visualize-the-state-of-education-in-colorado/data?select=2010_1YR_3YR_change.csv): Student performance, college-readiness, and demographic information by school for the years 2010-2012 
2. [School District Revenues and Expenditures | CDE (state.co.us)](http://www.cde.state.co.us/cdefinance/revexp): Revenue and expenditures by program and district for the years 2010-2012
3. [SAIPE Datasets (census.gov)](https://www.census.gov/programs-surveys/saipe/data/datasets.html): The number of students whose families below the poverty level by district for the years 2010-2012

#### Outline
Describe how the overall performance of the state has changed over the past years and break it down by districts, school level, type of school using visualization
Compare the number of students in poverty, the number of students receiving discounted lunches, and the amount of money being spent in support programs.
Examine the performance of students in minorities to make recommendations of outreach programs.
Demonstrate evidence for the effect of expenditures as well as their types on performance with hypothesis testing
Use clustering to compare performance, demographic information, or geographic location

----
##### Special thanks
![ColoradoSchoolGrades](https://scontent.fslc3-2.fna.fbcdn.net/v/t39.30808-6/305205208_158321073535107_6016190664016506602_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=cuVGXWEetJoAX_Mb3Zv&_nc_ht=scontent.fslc3-2.fna&oh=00_AfAyqm3EYIn1OOm87C5m0BTX8AWsEgmC_AoPSBTUmdR5VA&oe=63EEBF99)



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
