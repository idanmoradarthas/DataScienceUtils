# Change Log
All notable changes to this project will be documented in this file.

## [1.1] - 2019-11-20
### Added
- added new method metrics::plot_precision_recall for plotting precision recall curve for binary and multi class
- added matplotlib testing for plot_precision_recall
### Changed
- metrics::plot_roc_curve renamed to plot_roc_curve_binary_class
### Fixed
- minor changes

## [1.0.4] - 2019-11-19
### Added
- added install_requires, python_requires and license in setup.py
- added changelog
### Changed
- add version dependencies to requirements
- changed tox virtualenv
- changed classifiers in setup.py
### Fixed
- fix tests for strings
- minor changes

## [1.0.3] - 2019-09-23
### Added

### Changed

### Fixed
- handle DeprecationWarning when using visualization_aids::draw_tree
- fix tests for strings

## [1.0.2] - 2018-12-03
### Added
- added method for feature visualization as visualization_aids::visualize_features
### Changed
- updated syntax for dropping index in preprocess::get_correlated_features
- updated documentation for new feature visualization in visualization_aids::visualize_features
### Fixed

## [1.0.1] - 2018-09-24
### Added
- added module preprocess
- documentation for modules metrics, preprocess, strings and visualization_aids
### Changed
 
### Fixed
- minor changes to setup.py

## [1.0] - 2018-09-23
### Added
- Initial release:
    - created the metrics, strings and visualization modules
### Changed
 
### Fixed