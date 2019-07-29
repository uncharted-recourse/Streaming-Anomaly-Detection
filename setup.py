from distutils.core import setup

setup(name='manatee',
    version='1.0.0',
    description='methods for anomaly notification against time-series evidence',
    packages=['manatee'],
    install_requires=[
        'numpy>=1.14.2',
        'pandas>=0.19.2',
        'rrcf @ git+https://github.com/NewKnowledge/rrcf@b527d425c212db19dce3446d63b2e011d47d94e5#egg=rrcf-0.3',
    ],
    include_package_data=True)
