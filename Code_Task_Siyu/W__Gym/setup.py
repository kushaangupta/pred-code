import setuptools

setuptools.setup(
    name='W_Gym',
    version = '0.0.0',
    description='siyu''s Gym functions',
    author='Siyu Wang',
    author_email='wangxsiyu@gmail.com',
    packages = setuptools.find_packages(),
    install_requires =['gym','numpy','pygame'],
    zip_safe=False
)