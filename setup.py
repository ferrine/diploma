from setuptools import setup, find_packages

from diploma import __version__

if __name__ == '__main__':
    setup(
            name='forecast',
            version=__version__,
            description='Bayesian anomaly detection',
            url='https://github.yandex-team.ru/ferres/diploma/',
            author='Maxim Kochurov',
            author_email='maxim.v.kochurov@gmail.com',
            install_requires=['six', 'pymc3', 'seaborn', 'lasagne'],
            packages=find_packages(),
    )
