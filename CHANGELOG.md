# [[0.12.0](https://github.com/yoshoku/rumale-svm/compare/v0.11.0...v0.12.0)]
- Fix the version specification of rumale-core gem.

# [[0.11.0](https://github.com/yoshoku/rumale-svm/compare/v0.10.0...v0.11.0)]
- Add Rumale::SVM::ClusteredSVC that is classifier with clustered support vector machine.

# 0.10.0
- Add Rumale::SVM::RandomRecursiveSVC that is classifier with random recursive support vector machine.
- Add type declaration files for RandomRecursiveSVC and LocallyLinearSVC.

# 0.9.0
- Add Rumale::SVM::LocallyLinearSVC that is classifier with locally linear support vector machine.

# 0.8.0
- Refactor to support the new Rumale API.

# 0.7.0
- Support for probabilistic outputs with Rumale::SVM::OneClassSVM.
- Update numo-libsvm depedency to v2.1 or higher.

# 0.6.0
- Update numo-libsvm and numo-liblinear depedency to v2.0 or higher.

# 0.5.1
- Refactor specs and config files.

# 0.5.0
- Add type declaration files.

# 0.4.0
- Add linear one-class support vector machine.

# 0.3.0
- Fix to raise error when calling prediction method before training model.
- Fix some config files.

# 0.2.0
- Supported the new Rumale's validation:
  - Fix to use new numeric validation for hyperparameter values.
  - Fix to use new array validation that accepts Ruby Array for samples, labels, and target values.

# 0.1.0
- First release.
