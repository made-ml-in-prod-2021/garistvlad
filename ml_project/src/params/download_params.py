from typing import List

from dataclasses import dataclass, field


@dataclass()
class DownloadParams:
    s3_filepath: str
    output_folder: str
    s3_bucket: str = field(default="madeawsbucket")
