FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD language_identification_stopwords.py /code/language_identification_stopwords.py
ADD model.joblib /code/model.joblib

ENTRYPOINT [ "python3", "/code/language_identification_stopwords.py" ]
