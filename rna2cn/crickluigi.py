import luigi
from luigi import Task, LocalTarget
import subprocess
import tempfile
import os

import bioluigi.cluster
from bioluigi.utils import CheckTargetNonEmpty
from bioluigi.slurm import SlurmExecutableTask, SlurmTask

import logging
logger = logging.getLogger('luigi-interface')

class ExecTask(luigi.Task, bioluigi.cluster.ClusterBase):
    
    def __init__(self, *args, **kwargs):
        self.run_locally = True
        super().__init__(*args, **kwargs)
        
    def on_failure(self, exception):
        err = self.format_log()
        self.clear_tmp()
        logger.info(err)
        super_retval = super().on_failure(exception)
        ret = err if super_retval is None else err + "\n" + super_retval
        return ret

    def on_success(self):
        err = self.format_log()
        self.clear_tmp()
        logger.info(err)
        super_retval = super().on_success()
        ret = err if super_retval is None else err + "\n" + super_retval
        return ret

    def work_script(self):
        """Override this an make it return the shell script to run"""
        pass
    
    def run(self):

        # Write the launch script to file
        self.launcher = os.path.join(tempfile.gettempdir(), self.task_id + ".sh")

        with open(self.launcher, 'w') as l:
            l.write(self.work_script())
        # Make executable
        os.chmod(self.launcher, os.stat(self.launcher).st_mode | 0o111)

        self.completedprocess = subprocess.run(self.launcher, 
                                               shell=True,
                                               stderr=subprocess.PIPE, 
                                               stdout=subprocess.PIPE, 
                                               universal_newlines=True)
                                               
        self.completedprocess.check_returncode()
        
def get_ext(path):
    '''Split path into base and extention, gracefully handling compressed extensions eg .gz'''
    base, ext1 = os.path.splitext(path)
    if ext1 == '.gz':
        base, ext2 = os.path.splitext(base)
        return base, ext2 + ext1
    else:
        return base, ext1
        
class Trimmomatic(CheckTargetNonEmpty, SlurmExecutableTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition = 'main'
        self.mem = 2000
        self.n_cpu = 4
         
    def output(self):
        base = os.path.dirname(self.input()[0].path)
        return [LocalTarget(os.path.join(base, "trimmed_1.fastq.gz")),
                LocalTarget(os.path.join(base, "trimmed_2.fastq.gz")),
                LocalTarget(os.path.join(base, "trimmomatic.txt"))]

    def work_script(self):
        return '''#!/bin/bash
               set -euo pipefail
               module load Java;
               
               trimmomatic='java -XX:+UseSerialGC -Xmx{mem}M -jar /home/dbunting/Trimmomatic-0.36/trimmomatic-0.36.jar'
               
               $trimmomatic PE {R1_in} {R2_in} \
                               -threads 4 \
                               -baseout {base}.fastq.gz \
                               ILLUMINACLIP:{adapters}:2:30:10:4 SLIDINGWINDOW:4:20 MINLEN:50 \
                               > {log}.temp 2>&1
               
               rm {base}_?U.fastq.gz
               mv {base}_1P.fastq.gz {R1_out}
               mv {base}_2P.fastq.gz {R2_out}
               mv {log}.temp {log}
               '''.format(log=self.output()[2].path,
                          R1_in=self.input()[0].path,
                          R2_in=self.input()[1].path,
                          adapters='/home/dbunting/Trimmomatic-0.36/adapters/TrueSeq.cat.fa',
                          R1_out=self.output()[0].path,
                          R2_out=self.output()[1].path,
                          mem=int(0.5*self.mem*self.n_cpu),
                          base=os.path.join(os.path.dirname(self.input()[0].path), "trimmed"))
                          
    def run(self, *args, **kwargs):
        super_ret = super().run(*args, **kwargs)
        with self.output()[-1].open('r') as f:
            dirname = os.path.dirname(self.input()[0].path)
            name = os.path.split(dirname)[1]
            log = f.read()
            log = log.replace(self.input()[0].path, 
                              os.path.join(dirname, name + ".fastq.gz"))
        with self.output()[-1].open('w') as f:
            f.write(log)
        return super_ret
                          
# CheckTargetNonEmpty won't work for bam files here 
class Kallisto(SlurmExecutableTask):
    '''NB you can't use both --threads and --pseudobam'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition = 'main'
        self.mem = 4000
        self.n_cpu = 2
        
    def output(self):
        base = os.path.dirname(self.input()[0].path)
        return {'hd5':LocalTarget(os.path.join(base, "kallisto", "abundance.h5")),
                'tsv':LocalTarget(os.path.join(base, "kallisto", "abundance.tsv")),
                'bam':LocalTarget(os.path.join(base, "kallisto", "pseudoalign.bam")),
                'log':LocalTarget(os.path.join(base, "kallisto", "kallisto.log"))}

    def work_script(self):
        return '''#!/bin/bash
               mkdir {outdir}/kallisto_temp
               set -euo pipefail
               module load SAMtools
               
               kallisto quant -i {index} \
                              -o {outdir}/kallisto_temp \
                              --pseudobam \
                              {R1} {R2} \
                              2> {outdir}/kallisto_temp/kallisto.log |
                              samtools view -b - > {outdir}/kallisto_temp/pseudoalign.bam
                              
               mv {outdir}/kallisto_temp/* {outdir}/kallisto              
               '''.format(R1=self.input()[0].path,
                          R2=self.input()[1].path,
                          index="/srv/shared/vanloo/rna2cn/references/GRCh38_cdna_ercc92",
                          outdir=os.path.dirname(self.input()[0].path))
                          
    def run(self, *args, **kwargs):
        super_ret = super().run(*args, **kwargs)
        with self.output()['log'].open('r') as f:
            dirname = os.path.dirname(self.input()[0].path)
            name = os.path.split(dirname)[1]
            log = f.read()
            log = log.replace(self.input()[0].path, 
                              os.path.join(dirname, name + ".fastq.gz"))
        with self.output()['log'].open('w') as f:
            f.write(log)
        return super_ret
        
        
class BAMtoFASTQ(CheckTargetNonEmpty, SlurmExecutableTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition = 'main'
        self.mem = 4000
        self.n_cpu = 1
        
    def output(self):
        return LocalTarget(get_ext(self.input().path)[0] + ".fastq.gz")

    def work_script(self):
        return '''#!/bin/bash
                    set -euo pipefail
                    module load SAMtools
                    
                    samtools view {input} |
                    perl -ne 'chomp; $line = $_;  @s = split("\t", $line); print "\@$s[0]\n$s[9]\n+\n$s[10]\n";' |
                    gzip -c > {output}.temp
                    
                    mv {output}.temp {output}
        '''.format(input=self.input().path,
                   output=self.output().path)
                   

class DeinterleaveBAM(CheckTargetNonEmpty, SlurmExecutableTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition = 'main'
        self.mem = 4000
        self.n_cpu = 1
        

    def output(self):
        return [LocalTarget(get_ext(self.input().path)[0] + "_1.fastq.gz"),
                LocalTarget(get_ext(self.input().path)[0] + "_2.fastq.gz")]

    def work_script(self):
        return '''#!/bin/bash
                    set -euo pipefail
                    
                    zcat {input} |
                    paste - - - - - - - - | 
                    tee >(cut -f 1-4 | tr "\\t" "\\n" | gzip -c > {R1}.temp) | 
                          cut -f 5-8 | tr "\\t" "\\n" | gzip -c > {R2}.temp
                                      
                    mv {R1}.temp {R1}
                    mv {R2}.temp {R2}
        '''.format(input=self.input().path,
                   R1=self.output()[0].path,
                   R2=self.output()[1].path)                   
                   
class SamtoolsDepth(CheckTargetNonEmpty, SlurmExecutableTask):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition = 'main'
        self.mem = 4000
        self.n_cpu = 1
    
    def output(self):
            return LocalTarget(get_ext(self.input().path)[0] + "_depth.tsv")
    
    def work_script(self):
        return '''#!/bin/bash
                    module load SAMtools
                    
                    samtools depth {input} > {output}.temp
     
                    mv {output}.temp {output}
                    '''.format(input=self.input().path,
                               output=self.output().path)
    
    
class AggregateKallisto(CheckTargetNonEmpty, luigi.Task):
    sample = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition = 'main'
        self.mem = 16000
        self.n_cpu = 1
        
    def output(self):
        return LocalTarget(os.path.join(self.base_dir, "kallisto_abundances.tsv"))
    
    def run(self):
        import pandas as pd
        from luigi.file import atomic_file

        in_dfs = {s: pd.read_table(inp['tsv'].path).drop(['length', 'eff_length'], axis=1).set_index('target_id') 
                  for s,inp in self.input().items()}
        catd = pd.concat(in_dfs, axis=1)
        
        af = atomic_file(self.output().path)
        catd.to_csv(af.tmp_path, sep='\t')
        af.move_to_final_destination()
        
        
class SamtoolsSort(SlurmExecutableTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the SLURM request params for this task
        self.mem = 8000
        self.n_cpu = 1
        self.partition = "main"

    def output(self):
        return LocalTarget(get_ext(self.input().path)[0] + '_sorted.bam')

    def work_script(self):
        return '''#!/bin/bash
               set -euo pipefail
               module load SAMtools
               samtools sort --output-fmt BAM -o {output}.temp {input}
               mv {output}.temp {output}
                '''.format(input=self.input().path,
                           output=self.output().path)

    
class FastQC(CheckTargetNonEmpty, SlurmExecutableTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the SLURM request params for this task
        self.mem = 10000
        self.n_cpu = 1
        self.partition = "main"

    def output(self):
        return LocalTarget(os.path.join(os.path.dirname(self.input()[0].path), 'fastqc_data.txt'))

    def work_script(self):
        return '''#!/bin/bash
                mkdir -p {temp_dir}
                set -euo pipefail
                
                cd {temp_dir}
                cat {R1} {R2} > {sample}.fastq.gz
                fastqc {sample}.fastq.gz -o ./ -t 1
                unzip {sample}_fastqc.zip

                mv {sample}_fastqc/fastqc_data.txt {output}
                rm -rf {temp_dir}
                '''.format(temp_dir=os.path.join('/tmp/dbunting/', self.sample, 'FastQC'),
                           R1=self.input()[0].path,
                           R2=self.input()[1].path,
                           sample=self.sample,
                           output=self.output().path)