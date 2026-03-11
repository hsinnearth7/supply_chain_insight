terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket         = "chaininsight-terraform-state"
    key            = "environments/prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "chaininsight-terraform-locks"
    encrypt        = true
  }
}

module "infrastructure" {
  source = "../../"

  aws_region  = "us-east-1"
  environment = "prod"

  # EKS — production sizing
  eks_instance_types = ["t3.xlarge"]
  eks_desired_size   = 3
  eks_min_size       = 2
  eks_max_size       = 10

  # RDS — production with multi-AZ
  rds_instance_class    = "db.r6g.large"
  rds_allocated_storage = 100
  db_username           = "chaininsight"
  db_password           = var.db_password

  # Redis — larger for production
  redis_node_type  = "cache.r6g.large"
  redis_num_nodes  = 1
}

variable "db_password" {
  description = "Database password for production environment"
  type        = string
  sensitive   = true
}

output "eks_cluster_name" {
  value = module.infrastructure.eks_cluster_name
}

output "eks_cluster_endpoint" {
  value = module.infrastructure.eks_cluster_endpoint
}

output "rds_endpoint" {
  value = module.infrastructure.rds_endpoint
}

output "redis_endpoint" {
  value = module.infrastructure.redis_endpoint
}

output "s3_data_bucket" {
  value = module.infrastructure.s3_data_bucket
}

output "s3_models_bucket" {
  value = module.infrastructure.s3_models_bucket
}
