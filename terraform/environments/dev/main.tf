terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket         = "chaininsight-terraform-state"
    key            = "environments/dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "chaininsight-terraform-locks"
    encrypt        = true
  }
}

module "infrastructure" {
  source = "../../"

  aws_region  = "us-east-1"
  environment = "dev"

  # EKS — smaller for dev
  eks_instance_types = ["t3.medium"]
  eks_desired_size   = 1
  eks_min_size       = 1
  eks_max_size       = 3

  # RDS — minimal for dev
  rds_instance_class    = "db.t3.micro"
  rds_allocated_storage = 10
  db_username           = "chaininsight"
  db_password           = var.db_password

  # Redis — single node for dev
  redis_node_type  = "cache.t3.micro"
  redis_num_nodes  = 1
}

variable "db_password" {
  description = "Database password for dev environment"
  type        = string
  sensitive   = true
}

output "eks_cluster_name" {
  value = module.infrastructure.eks_cluster_name
}

output "rds_endpoint" {
  value = module.infrastructure.rds_endpoint
}

output "redis_endpoint" {
  value = module.infrastructure.redis_endpoint
}
