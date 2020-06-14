# Change Log
All notable changes to this project will be documented in this file.

## [1.6] - 2020-06-10
### Added
- visualization_aids::visualize_correlations function that plot heatmap of features' correlations.
- visualization_aids::plot_features_relationship function that plot the shared distribution of two features.
- documentation for the new methods.
### Changed
- requirements dependencies.
- visualization_aids::generate_decision_paths moved to xai::generate_decision_paths.
- visualization_aids::visualize_features changed parameter name frame to data.
### Fixed
- documentation fixes.
- tests fixes.
- minor changes

## [1.5] - 2020-01-08
### Added
 - strings::extract_significant_terms_from_subset function that extract significant terms from a data subset like 
 elasticsearch significant_text aggregation.
 - automated testing and code coverage.
 - deployment to conda.
### Changed
- confusion matrix image in read me converted to a github link.
- strings::append_tags_to_frame added parameters max_features, lowercase and tokenizer.
- visualization_aids::plot_metric_growth_per_labeled_instances moved to metrics.
### Fixed
- minor changes

## [1.4.1] - 2019-12-26
### Added

### Changed

### Fixed
- minor changes

## [1.4] - 2019-12-26
### Added
- visualization_aids::plot_metric_growth_per_labeled_instances function that plot given metric change where the amount 
of labeled instances increase.
- visualization_aids::print_decision_paths can now receives a char for indentation markings.
- metrics::plot_confusion_matrix receives more seaborn parameters for better control over plotting.
- visualization_aids::draw_dot_data function that plot Graphviz's dot data.
### Changed
- package name renamed to ```data_science_utils```.
- visualization_aids::print_decision_paths default indent char changed from "  " to "\t".
- rewrite README.md
- revamp documentation with ```read the docs``` theme.
### Fixed
- package description and keywords
- minor changes

## [1.3] - 2019-12-10
### Added
- visualization_aids::visualize_features added parameters: features: list of feature to visualize, 
num_columns: number of columns in the grid, and remove_na: True to ignore NA values when plotting; False otherwise.
### Changed
- visualization_aids::draw_tree changed signature to matplotlib coding style 
(see [matplotlib Usage Guide](https://matplotlib.org/tutorials/introductory/usage.html#coding-styles)).
- all drawing method now return matplotlib.axes.Axes instead of matplotlib.pyplot.Figure.
### Fixed
- Revert import change of sklearn.tree.tree to sklearn.tree due to FutureWarning.


## [1.2] - 2019-12-09
### Added
- visualization_aids::print_decision_paths a method that converts decision tree to a python function.
### Changed
- visualization_aids::draw_tree parameter features_names changed to feature_names.
- visualization_aids::draw_tree parameters feature_names and class_names received default value.
### Fixed
- Changed import of sklearn.tree.tree to sklearn.tree due to FutureWarning.

## [1.1] - 2019-11-20
### Added
- added matplotlib testing
### Changed
- removed metrics::plot_precision_recall and metrics::plot_roc_curve due duplication with Yellowbrick package
- changed metrics::print_confusion_matrix to plot_confusion_matrix which returns a matplotlib figure
- visualization_aids now do not require ipython
- visualization_aids returns matplotlib figure objects
- metrics returns matplotlib figure objects
### Fixed
- docs
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