"""Register an already-uploaded ZIP using OCI's large-artifact export API.

Run explicitly after reviewing the artifact. Does not provision a deployment.
"""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="Object Storage URI of exported model ZIP")
    parser.add_argument("--compartment-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--display-name", default="Tabula Nova 8B")
    parser.add_argument("--auth", choices=("resource_principal", "instance_principal", "config"), default="resource_principal")
    parser.add_argument("--profile", default="DEFAULT")
    args = parser.parse_args(argv)
    import oci
    from tabula_nova.oci_storage import parse_uri
    bucket, namespace, key = parse_uri(args.uri)
    if not key.endswith(".zip"):
        raise ValueError("model artifact must be a ZIP")
    if args.auth == "config":
        config = oci.config.from_file(profile_name=args.profile)
        config["region"] = args.region
        service = oci.data_science.DataScienceClient(config)
    else:
        signer = (oci.auth.signers.get_resource_principals_signer() if args.auth == "resource_principal"
                  else oci.auth.signers.InstancePrincipalsSecurityTokenSigner())
        service = oci.data_science.DataScienceClient({"region": args.region}, signer=signer)
    model = service.create_model(oci.data_science.models.CreateModelDetails(
        compartment_id=args.compartment_id, project_id=args.project_id, display_name=args.display_name,
        description="Tabula-derived 8B research model; consult artifact manifest for training provenance.")).data
    response = service.export_model_artifact(model.id, oci.data_science.models.ExportModelArtifactDetails(
        artifact_export_details=oci.data_science.models.ArtifactExportDetailsObjectStorage(
            artifact_source_type="ORACLE_OBJECT_STORAGE", namespace=namespace,
            source_bucket=bucket, source_object_name=key, source_region=args.region)))
    print(json.dumps({"model_id": model.id, "work_request_id": response.headers.get("opc-work-request-id"),
                      "status": "artifact transfer submitted; verify work request succeeds before deploying"}, indent=2))


if __name__ == "__main__":
    main()
