# spaces.py
import os
import uuid
from urllib.parse import urljoin

import boto3


class SpacesClient:
    """
    A clean class-based helper for DigitalOcean Spaces.
    Provides upload + public URL generation.
    """

    def __init__(
        self,
        key: str = None,
        secret: str = None,
        bucket: str = None,
        region: str = None,
        endpoint: str = None,
        cdn_base: str = None,
    ):
        # Defaults from environment
        self.key = key or os.getenv("SPACES_KEY")
        self.secret = secret or os.getenv("SPACES_SECRET")
        self.bucket = bucket or os.getenv("SPACES_BUCKET")
        self.region = region or os.getenv("SPACES_REGION", "nyc3")
        self.endpoint = endpoint or os.getenv(
            "SPACES_ENDPOINT",
            f"https://{self.bucket}.{self.region}.digitaloceanspaces.com",
        )
        self.cdn_base = cdn_base or os.getenv("SPACES_CDN_BASE")

        if not self.key or not self.secret or not self.bucket:
            raise RuntimeError("Spaces configuration missing (KEY/SECRET/BUCKET).")

        # Lazy-loaded boto3 client
        self._client = None

    # --------------------------- #
    # Internal helpers
    # --------------------------- #

    @property
    def client(self):
        """Lazy-initialize the boto client."""
        if self._client is None:
            session = boto3.session.Session()
            self._client = session.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint,
                aws_access_key_id=self.key,
                aws_secret_access_key=self.secret,
            )
        return self._client

    @staticmethod
    def _generate_key(filename: str) -> str:
        """Generates a unique object key (path) for upload."""
        ext = os.path.splitext(filename or "")[1] or ".jpg"
        return f"rocks/{uuid.uuid4().hex}{ext}"

    def public_url(self, key: str) -> str:
        """
        Build a public URL for an uploaded object.
        Prefer CDN domain when available.
        """
        if self.cdn_base:
            return urljoin(self.cdn_base.rstrip("/") + "/", key.lstrip("/"))

        # endpoint already points at https://{bucket}.{region}.digitaloceanspaces.com
        return f"{self.endpoint.rstrip('/')}/{key.lstrip('/')}"

    # --------------------------- #
    # Public API
    # --------------------------- #

    def upload_bytes(
        self,
        data: bytes,
        filename: str,
        content_type: str = "image/jpeg",
        acl: str = "public-read",
    ) -> str:
        """
        Upload raw bytes to Spaces and return the public URL.
        """
        key = self._generate_key(filename)

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ACL=acl,
            ContentType=content_type,
        )

        return self.public_url(key)
