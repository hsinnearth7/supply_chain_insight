output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.database_name
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.redis.endpoint
}

output "s3_data_bucket" {
  description = "S3 bucket for data storage"
  value       = module.s3.data_bucket_name
}

output "s3_models_bucket" {
  description = "S3 bucket for model artifacts"
  value       = module.s3.models_bucket_name
}

output "database_url" {
  description = "Full PostgreSQL connection URL"
  value       = "postgresql://${var.db_username}:****@${module.rds.endpoint}:5432/chaininsight"
  sensitive   = true
}

output "redis_url" {
  description = "Full Redis connection URL"
  value       = "redis://${module.redis.endpoint}:6379/0"
}
