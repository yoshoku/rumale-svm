# frozen_string_literal: true

lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'rumale/svm/version'

Gem::Specification.new do |spec|
  spec.name          = 'rumale-svm'
  spec.version       = Rumale::SVM::VERSION
  spec.authors       = ['yoshoku']
  spec.email         = ['yoshoku@outlook.com']

  spec.summary       = <<~MSG
    Rumale::SVM provides support vector machine algorithms using LIBSVM and LIBLINEAR with Rumale interface.
  MSG
  spec.description = <<~MSG
    Rumale::SVM provides support vector machine algorithms using LIBSVM and LIBLINEAR with Rumale interface.
  MSG
  spec.homepage = 'https://github.com/yoshoku/rumale-svm'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = 'https://github.com/yoshoku/rumale-svm'
  spec.metadata['changelog_uri'] = 'https://github.com/yoshoku/rumale-svm/blob/main/CHANGELOG.md'
  spec.metadata['documentation_uri'] = 'https://yoshoku.github.io/rumale-svm/doc/'
  spec.metadata['rubygems_mfa_required'] = 'true'

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features|sig-deps)/}) }
                     .select { |f| f.match(/\.(?:rb|rbs|md|txt)$/) }
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  spec.add_dependency 'numo-liblinear', '~> 2.0'
  spec.add_dependency 'numo-libsvm', '~> 2.1'
  spec.add_dependency 'rumale-core', '>= 0.24'
end
