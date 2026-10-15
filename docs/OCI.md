# Oracle Cloud Infrastructure workflow

This integration targets **OCI Data Science Jobs, Object Storage and Model Catalog/Model Deployment**. Oracle Machine Learning inside an Autonomous Database is a different execution target. No tenancy resources or paid jobs are created by inspection, packaging or export.

Oracle requires a model artifact with `score.py` and `runtime.yaml`, with runtime metadata pointing to a compatible published inference environment. Nova writes both, includes the inference code at the artifact root, serializes BF16 weights into safetensor shards, and creates a ZIP64 archive with no optimizer state. [Oracle runtime schema](https://docs.oracle.com/en-us/iaas/Content/data-science/using/model_runtime_yaml.htm), [model artifacts](https://docs.oracle.com/iaas/Content/data-science/using/models-prepare-artifact.htm).

## Prepare the GPU environment

Use a published custom Conda environment containing Python 3.11, a CUDA-compatible PyTorch 2.8 build, `tokenizers`, `safetensors`, NumPy, tqdm and OCI SDK. Install this project with `python -m pip install -e '.[oci]'` in notebook development. The job archive uses `PYTHONPATH` and needs its dependencies preinstalled; `requirements.txt` in an exported model is an inventory, not an instruction that OCI automatically executes.

The inference environment slug and Object Storage path must refer to the environment you actually published. Keep its Python version aligned with `--python-version` at export. The runtime file is JSON-formatted YAML, which is valid YAML and follows Oracle's version 3.0 schema.

The recommended starting training topology is one node with eight BF16-capable GPUs and sufficient memory per GPU; provision memory conservatively, then measure a three-update preflight. The calculated persistent state alone is roughly 15 GiB per GPU at eight ranks. Use 80 GiB-class GPUs for headroom where available. Avoid treating this as a demonstrated throughput or cost estimate. For single-GPU inference, plan at least 24 GiB GPU memory and preferably 48 GiB headroom, plus enough host memory to load the weights. Shape availability depends on your region and quotas.

## Transfer data and submit the job

```bash
python -m tabula_nova upload --path prepared/mixed \
  --uri oci://MY_BUCKET@MY_NAMESPACE/nova/data/mixed/ --auth config
python -m tabula_nova pack-job --output dist/tabula_nova_8b.zip
```

Choose a Data Science GPU job with that source ZIP and set `JOB_RUN_ENTRYPOINT=tabula_nova_8b/oci/train.sh`. The ZIP root matches its filename, as [Oracle's job artifact documentation](https://docs.oracle.com/en-us/iaas/Content/data-science/using/jobs-artifact.htm) requires. Copy the settings in `oci/environment.example.json`, replacing paths with your mounted data and persistent checkpoint paths. Download the corpus beforehand from a notebook or setup process:

```bash
python -m tabula_nova download \
  --uri oci://MY_BUCKET@MY_NAMESPACE/nova/data/mixed/ \
  --path /mnt/nova/prepared --auth resource_principal
```

Do not download into an existing directory. Training verifies dataset hashes. Resource principal authentication uses the job/notebook dynamic group; local transfers can use `--auth config --profile DEFAULT`. Instance principal authentication is also available for Compute instances. Configure Object Storage read/write permissions for the corresponding principal. No keys are embedded in archives.

For preflight, set `NOVA_TRAIN_CONFIG` to a copy of `configs/train-oci.json` with `max_steps=3`, `warmup_steps=0`, `eval_every=3` and `save_every=3`; use a separate output directory. Measure GPU peak memory, loss, finite gradients, throughput, DCP checkpoint creation and restart. Test the `8b-attnres` variant separately. Resume the real run with `NOVA_RESUME` set to a complete step directory; keep its original training schedule unchanged.

`NOVA_OUTPUT_URI` optionally uploads the output tree when training finishes. UploadManager performs multipart transfers for large files. Omit this variable if checkpoint storage is already persistent/shared. Interrupted jobs preserve completed checkpoints on the mounted volume; the helper does not promise automatic cloud synchronization during interruption.

Multinode runs require `NOVA_NNODES`, each node's `NOVA_NODE_RANK`, rank zero's private `NOVA_MASTER_ADDR`, and the same `NOVA_GPUS_PER_NODE`. Supply these from your cluster configuration. All nodes must access the same data/checkpoint filesystem, and ranks must be allowed to communicate over the selected rendezvous and NCCL network ports. The helper does not infer ranks from OCI's eventually consistent cluster metadata or configure RDMA/NCCL device selection. Oracle documents [distributed jobs and shared filesystem support](https://docs.oracle.com/en-us/iaas/Content/data-science/using/jobs-distributed-jobs.htm).

## Export the trained checkpoint

```bash
python -m tabula_nova export-oci \
  --checkpoint /mnt/nova/runs/8b/step-00000100 \
  --output exports/nova \
  --env-slug MY_PUBLISHED_GPU_ENV \
  --env-path oci://MY_BUCKET@MY_NAMESPACE/conda/MY_ENV.tar.gz \
  --python-version 3.11 \
  --shard-size-mib 1024 \
  --zip dist/nova-model.zip
```

Local `latest.pt` checkpoints are accepted too. For local checkpoints the exporter memory-maps the source and converts model tensors one shard at a time. For distributed checkpoints it requests only the current model shard from DCP, avoiding full-model consolidation and optimizer allocation. That partial DCP path needs hardware-backed confirmation. Allow approximately 30 GiB free disk for the BF16 artifact and ZIP, in addition to training checkpoints. Exporting a checkpoint does not improve its learned quality.

The artifact contains `config.json`, `tokenizer.json`, `model.safetensors.index.json`, weight shards, the Nova inference package, `score.py`, `runtime.yaml`, requirements and a SHA-256 provenance manifest. Verify it locally:

```bash
python -m tabula_nova generate --artifact exports/nova --device cuda \
  --prompt 'Compute 3 + 4.' --max-new-tokens 64
python -m tabula_nova benchmark --artifact exports/nova --device cuda \
  --suite template --mode answer --context-limit 1024
```

## Register through Object Storage

```bash
python -m tabula_nova upload --path dist/nova-model.zip \
  --uri oci://MY_BUCKET@MY_NAMESPACE/nova/models/nova-model.zip --auth config
python oci/register_model.py \
  --uri oci://MY_BUCKET@MY_NAMESPACE/nova/models/nova-model.zip \
  --compartment-id ocid1.compartment.REPLACE \
  --project-id ocid1.datascienceproject.REPLACE \
  --region YOUR_REGION --auth config
```

The registration helper creates a catalog entry and submits Oracle's `export_model_artifact` operation from the uploaded Object Storage ZIP to the service bucket. It returns the model OCID and work request ID; confirm the work request succeeds before deploying. The Data Science/Object Storage services need access to the source bucket. [Oracle's large-artifact transfer procedure and policies](https://docs.oracle.com/en-us/iaas/Content/data-science/using/large-model-artifact-export.htm), [official SDK example](https://docs.oracle.com/en-us/iaas/tools/python-sdk-examples/latest/datascience/export_model_artifact.py.html).

Oracle's current catalog overview documents support for artifacts up to 400 GB, while older direct-upload pages still mention a 6 GB limit. The roughly 16 GB Nova artifact deliberately uses the large-artifact transfer API rather than Console/direct upload. Verify service limits in your tenancy. [Current catalog documentation](https://docs.oracle.com/en-us/iaas/Content/data-science/using/models_saving_catalog.htm).

Create the Model Deployment from the registered model with a compatible GPU shape and the published runtime environment. Start with one scoring worker and scale with separate deployment replicas; multiple worker processes each allocate their own model. Nova serializes scoring within each worker to bound concurrent cache memory. The supplied interface accepts:

```json
{"prompt": "Compute 3 + 4.", "max_new_tokens": 128, "temperature": 0.0, "top_k": 40}
```

It returns `generated_text` and `generated_tokens`. Requests must fit prompt plus generation into the trained context; generation is bounded to 1,024 output tokens. This is a synchronous custom decoder endpoint without streaming or continuous batching. Tokenizer/weights are bundled, so inference makes no model-hub requests.
