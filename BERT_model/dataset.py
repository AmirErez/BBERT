from torch.utils.data import IterableDataset, get_worker_info
from Bio import SeqIO
import re
import gzip


class FastqIterableDataset(IterableDataset):
    def __init__(self, file_path, chunk_size=1000, max_reads=None, max_length=None):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.max_reads = max_reads
        self.max_length = max_length  # Add max_length as an argument

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0
        num_workers = 1

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if self.file_path.endswith(('.fasta.gz', '.fna.gz')):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = "fasta"
        elif self.file_path.endswith('.fastq.gz'):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = "fastq"
        elif self.file_path.endswith('.gz'):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = "fastq"  # default to fastq for other .gz files
        elif self.file_path.endswith(('.fasta', '.fna')):
            open_func = lambda x: open(x, 'rt')
            file_format = "fasta"
        elif self.file_path.endswith('.fastq'):
            open_func = lambda x: open(x, 'rt')
            file_format = "fastq"
        else:
            raise ValueError("Unsupported file format. Supported: .fasta, .fna, .fastq, .fasta.gz, .fna.gz, .fastq.gz")

        with open_func(self.file_path) as handle:
            for i, chunk in enumerate(self._yield_chunks(handle, file_format)):
                if (i % num_workers) != worker_id:
                    continue
                for sample in chunk:
                    yield sample

    def _yield_chunks(self, handle, file_format):
        chunk = []
        read_count = 0
        for record in SeqIO.parse(handle, file_format):
            chunk.append({
                'id': str(record.id),
                'seq': str(record.seq)[:self.max_length] if self.max_length else str(record.seq)
            })
            
            read_count += 1
            if len(chunk) == self.chunk_size:
                yield chunk
                chunk = []
            if self.max_reads and read_count == self.max_reads:
                yield chunk    
                return  # Use return instead of break to exit the method
        if chunk:  # Yield any remaining chunk
            yield chunk

    def __len__(self):
        if self.max_reads is not None:
            return self.max_reads

        # Count all reads in the file (expensive for large files)
        if self.file_path.endswith(('.fasta.gz', '.fna.gz')):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = "fasta"
        elif self.file_path.endswith('.fastq.gz'):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = "fastq"
        elif self.file_path.endswith('.gz'):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = "fastq"  # default to fastq for other .gz files
        elif self.file_path.endswith(('.fasta', '.fna')):
            open_func = lambda x: open(x, 'rt')
            file_format = "fasta"
        elif self.file_path.endswith('.fastq'):
            open_func = lambda x: open(x, 'rt')
            file_format = "fastq"
        else:
            raise ValueError("Unsupported file format. Supported: .fasta, .fna, .fastq, .fasta.gz, .fna.gz, .fastq.gz")

        with open_func(self.file_path) as handle:
            return sum(1 for _ in SeqIO.parse(handle, file_format))
        
    def get_stats(self):
        """
        Returns:
            - seq_lens: list of int (lengths of each sequence)
            - total_reads: int (number of sequences)
        Caches results to avoid re-parsing.
        """
        if hasattr(self, "_seq_lens") and self._seq_lens is not None:
            return self._seq_lens, self._total_reads

        if self.file_path.endswith(('.fasta.gz', '.fna.gz')):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = 'fasta'
        elif self.file_path.endswith('.fastq.gz'):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = 'fastq'
        elif self.file_path.endswith('.gz'):
            open_func = lambda x: gzip.open(x, 'rt')
            file_format = 'fastq'  # default to fastq for other .gz files
        elif self.file_path.endswith(('.fasta', '.fna')):
            open_func = lambda x: open(x, 'rt')
            file_format = 'fasta'
        elif self.file_path.endswith('.fastq'):
            open_func = lambda x: open(x, 'rt')
            file_format = 'fastq'
        else:
            raise ValueError("Unsupported file format. Supported: .fasta, .fna, .fastq, .fasta.gz, .fna.gz, .fastq.gz")

        seq_lens = []
        with open_func(self.file_path) as handle:
            for i, record in enumerate(SeqIO.parse(handle, file_format)):
                seq_lens.append(len(record.seq))
                if self.max_reads is not None and i + 1 >= self.max_reads:
                    break

        total_reads = len(seq_lens)

        self._seq_lens = seq_lens
        self._total_reads = total_reads

        return seq_lens, total_reads
