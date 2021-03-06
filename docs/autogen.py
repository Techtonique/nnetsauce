# -*- coding: utf-8 -*-
import pathlib
import shutil
import keras_autodoc

PAGES = {    
    'documentation/classifiers.md': [
        'nnetsauce.AdaBoostClassifier',
        'nnetsauce.AdaBoostClassifier.fit',
        'nnetsauce.AdaBoostClassifier.predict',
        'nnetsauce.AdaBoostClassifier.predict_proba',
        'nnetsauce.AdaBoostClassifier.score',
        'nnetsauce.CustomClassifier',
        'nnetsauce.CustomClassifier.fit',
        'nnetsauce.GLMClassifier',
        'nnetsauce.GLMClassifier.fit',
        'nnetsauce.MultitaskClassifier',
        'nnetsauce.MultitaskClassifier.fit',
        'nnetsauce.RandomBagClassifier',
        'nnetsauce.RandomBagClassifier.fit',
        'nnetsauce.Ridge2Classifier',
        'nnetsauce.Ridge2Classifier.fit',
        'nnetsauce.Ridge2MultitaskClassifier',
        'nnetsauce.Ridge2MultitaskClassifier.fit',
        'nnetsauce.RandomBagClassifier',
        'nnetsauce.RandomBagClassifier.fit',
    ],
    'documentation/regressors.md': [
       'nnetsauce.BaseRegressor',
       'nnetsauce.BaseRegressor.fit',
       'nnetsauce.BaseRegressor.predict',
       'nnetsauce.BaseRegressor.score',
        'nnetsauce.BayesianRVFLRegressor',
        'nnetsauce.BayesianRVFLRegressor.fit',
        'nnetsauce.BayesianRVFL2Regressor',
        'nnetsauce.BayesianRVFL2Regressor.fit',        
        'nnetsauce.CustomRegressor',
        'nnetsauce.CustomRegressor.fit',
        'nnetsauce.GLMRegressor',
        'nnetsauce.GLMRegressor.fit',
        'nnetsauce.Ridge2Regressor',
        'nnetsauce.Ridge2Regressor.fit',
    ],    
    'documentation/time_series.md': [
        'nnetsauce.MTS',
        'nnetsauce.MTS.fit',
        'nnetsauce.MTS.predict',
    ]
}

nnetsauce_dir = pathlib.Path(__file__).resolve().parents[1]


def generate(dest_dir):
    template_dir = nnetsauce_dir / 'docs' / 'templates'

    doc_generator = keras_autodoc.DocumentationGenerator(
        PAGES,
        'https://github.com/Techtonique/nnetsauce',
        template_dir,
        #nnetsauce_dir / 'examples'
    )
    doc_generator.generate(dest_dir)

    readme = (nnetsauce_dir / 'README.md').read_text()
    index = (template_dir / 'index.md').read_text()
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    (dest_dir / 'index.md').write_text(index, encoding='utf-8')
    shutil.copyfile(nnetsauce_dir / 'CONTRIBUTING.md',
                    dest_dir / 'contributing.md')
    #shutil.copyfile(nnetsauce_dir / 'docs' / 'extra.css',
    #                dest_dir / 'extra.css')


if __name__ == '__main__':
    generate(nnetsauce_dir / 'docs' / 'sources')