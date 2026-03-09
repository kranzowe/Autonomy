from setuptools import find_packages, setup

package_name = 'targeting'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='vpietropaolo1998@gmail.com',
    description='Topic producing angle and distance from robot given bounding box of object with known dimensions',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'paper_targeting = targeting.paper_targeting:main'
        ],
    },
)
