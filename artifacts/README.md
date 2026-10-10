# Model artifact provenance
legacy-manifest.json records historical Git blob identities and the source commit.
A Git blob SHA is an identity checksum, not a model quality guarantee.
Keep the corresponding preprocessing/class manifest with each model export.
New datasets, checkpoints, event logs and training runs should be stored outside
normal Git (releases/LFS or controlled data storage), with SHA-256, environment,
training configuration, dataset version/split and evaluation metrics recorded.
Historical artifacts remain available until external storage and redistribution
permissions are verified. No clinical or held-out accuracy is inferred here.
