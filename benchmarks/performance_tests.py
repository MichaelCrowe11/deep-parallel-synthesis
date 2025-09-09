"""
Performance benchmarks for Deep Parallel Synthesis
Tests API latency, throughput, and resource utilization
"""

import asyncio
import time
import statistics
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import concurrent.futures

import httpx
import psutil
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float  # requests per second
    cpu_usage: float
    memory_usage: float
    error_rate: float
    timestamp: str


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    base_url: str = "http://localhost:8000"
    api_key: str = "dps_test_key"
    num_requests: int = 100
    concurrent_users: int = 10
    timeout: float = 30.0
    warm_up_requests: int = 10


class PerformanceBenchmark:
    """
    Performance benchmark suite for DPS API
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # Test prompts for different complexity levels
        self.test_prompts = {
            "simple": [
                "What is 2+2?",
                "Name three colors.",
                "What day is today?"
            ],
            "medium": [
                "Explain the water cycle in detail.",
                "What are the causes of climate change?",
                "How does photosynthesis work?"
            ],
            "complex": [
                "Analyze the economic implications of artificial intelligence on job markets.",
                "Explain quantum entanglement and its applications in quantum computing.",
                "Discuss the philosophical implications of consciousness in artificial intelligence."
            ]
        }
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks"""
        logger.info("Starting DPS Performance Benchmarks")
        
        # Run benchmarks
        await self.benchmark_api_latency()
        await self.benchmark_concurrent_load()
        await self.benchmark_batch_processing()
        await self.benchmark_different_complexities()
        await self.benchmark_model_loading()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    async def benchmark_api_latency(self):
        """Benchmark API latency for single requests"""
        logger.info("Running API latency benchmark...")
        
        latencies = []
        successful = 0
        failed = 0
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            # Warm up
            for _ in range(self.config.warm_up_requests):
                await self._make_reasoning_request(client, "Warm up request")
            
            # Measure CPU and memory before
            cpu_start = psutil.cpu_percent()
            memory_start = psutil.virtual_memory().percent
            
            start_time = time.time()
            
            for i in range(self.config.num_requests):
                prompt = self.test_prompts["simple"][i % len(self.test_prompts["simple"])]
                
                request_start = time.time()
                success = await self._make_reasoning_request(client, prompt)
                request_end = time.time()
                
                latency = request_end - request_start
                latencies.append(latency)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{self.config.num_requests} requests")
            
            total_time = time.time() - start_time
            
            # Measure CPU and memory after
            cpu_end = psutil.cpu_percent()
            memory_end = psutil.virtual_memory().percent
        
        # Calculate metrics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = successful / total_time
        error_rate = failed / (successful + failed) * 100
        
        result = BenchmarkResult(
            name="API Latency",
            total_requests=self.config.num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=throughput,
            cpu_usage=(cpu_start + cpu_end) / 2,
            memory_usage=(memory_start + memory_end) / 2,
            error_rate=error_rate,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        logger.info(f"API Latency benchmark completed: {throughput:.2f} RPS")
    
    async def benchmark_concurrent_load(self):
        """Benchmark concurrent load handling"""
        logger.info(f"Running concurrent load benchmark with {self.config.concurrent_users} users...")
        
        latencies = []
        successful = 0
        failed = 0
        
        async def worker(client: httpx.AsyncClient, requests_per_user: int) -> List[float]:
            worker_latencies = []
            for i in range(requests_per_user):
                prompt = self.test_prompts["medium"][i % len(self.test_prompts["medium"])]
                
                request_start = time.time()
                success = await self._make_reasoning_request(client, prompt)
                request_end = time.time()
                
                worker_latencies.append(request_end - request_start)
                
                if not success:
                    nonlocal failed
                    failed += 1
                else:
                    nonlocal successful
                    successful += 1
            
            return worker_latencies
        
        # Calculate requests per user
        requests_per_user = self.config.num_requests // self.config.concurrent_users
        
        # Measure system resources
        cpu_start = psutil.cpu_percent()
        memory_start = psutil.virtual_memory().percent
        
        start_time = time.time()
        
        # Run concurrent workers
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            tasks = [
                worker(client, requests_per_user)
                for _ in range(self.config.concurrent_users)
            ]
            
            worker_results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Measure system resources after
        cpu_end = psutil.cpu_percent()
        memory_end = psutil.virtual_memory().percent
        
        # Flatten latencies
        for worker_latencies in worker_results:
            latencies.extend(worker_latencies)
        
        # Calculate metrics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = successful / total_time
        error_rate = failed / (successful + failed) * 100 if (successful + failed) > 0 else 0
        
        result = BenchmarkResult(
            name="Concurrent Load",
            total_requests=successful + failed,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=throughput,
            cpu_usage=(cpu_start + cpu_end) / 2,
            memory_usage=(memory_start + memory_end) / 2,
            error_rate=error_rate,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        logger.info(f"Concurrent load benchmark completed: {throughput:.2f} RPS")
    
    async def benchmark_batch_processing(self):
        """Benchmark batch processing performance"""
        logger.info("Running batch processing benchmark...")
        
        batch_size = 10
        num_batches = self.config.num_requests // batch_size
        
        latencies = []
        successful = 0
        failed = 0
        
        async with httpx.AsyncClient(timeout=self.config.timeout * 2) as client:
            cpu_start = psutil.cpu_percent()
            memory_start = psutil.virtual_memory().percent
            
            start_time = time.time()
            
            for batch_num in range(num_batches):
                # Create batch of prompts
                prompts = [
                    self.test_prompts["simple"][(batch_num * batch_size + i) % len(self.test_prompts["simple"])]
                    for i in range(batch_size)
                ]
                
                batch_data = {
                    "prompts": prompts,
                    "common_params": {
                        "max_depth": 2,
                        "num_chains": 4,
                        "temperature": 0.7
                    }
                }
                
                request_start = time.time()
                
                try:
                    response = await client.post(
                        f"{self.config.base_url}/reason/batch",
                        json=batch_data,
                        headers={"Authorization": f"Bearer {self._get_test_token()}"}
                    )
                    
                    request_end = time.time()
                    latency = request_end - request_start
                    latencies.append(latency)
                    
                    if response.status_code == 200:
                        result = response.json()
                        successful += result.get("successful", 0)
                        failed += result.get("failed", 0)
                    else:
                        failed += batch_size
                        
                except Exception as e:
                    logger.warning(f"Batch request failed: {e}")
                    failed += batch_size
                    latencies.append(self.config.timeout)
            
            total_time = time.time() - start_time
            
            cpu_end = psutil.cpu_percent()
            memory_end = psutil.virtual_memory().percent
        
        # Calculate metrics
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = 0
        
        throughput = successful / total_time if total_time > 0 else 0
        error_rate = failed / (successful + failed) * 100 if (successful + failed) > 0 else 0
        
        result = BenchmarkResult(
            name="Batch Processing",
            total_requests=successful + failed,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=throughput,
            cpu_usage=(cpu_start + cpu_end) / 2,
            memory_usage=(memory_start + memory_end) / 2,
            error_rate=error_rate,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        logger.info(f"Batch processing benchmark completed: {throughput:.2f} RPS")
    
    async def benchmark_different_complexities(self):
        """Benchmark performance with different prompt complexities"""
        complexities = ["simple", "medium", "complex"]
        
        for complexity in complexities:
            logger.info(f"Running {complexity} complexity benchmark...")
            
            latencies = []
            successful = 0
            failed = 0
            
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                cpu_start = psutil.cpu_percent()
                memory_start = psutil.virtual_memory().percent
                
                start_time = time.time()
                
                requests_to_run = min(self.config.num_requests, 30)  # Limit for complex prompts
                
                for i in range(requests_to_run):
                    prompts = self.test_prompts[complexity]
                    prompt = prompts[i % len(prompts)]
                    
                    request_start = time.time()
                    success = await self._make_reasoning_request(client, prompt)
                    request_end = time.time()
                    
                    latency = request_end - request_start
                    latencies.append(latency)
                    
                    if success:
                        successful += 1
                    else:
                        failed += 1
                
                total_time = time.time() - start_time
                
                cpu_end = psutil.cpu_percent()
                memory_end = psutil.virtual_memory().percent
            
            # Calculate metrics
            avg_latency = statistics.mean(latencies) if latencies else 0
            min_latency = min(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            p95_latency = np.percentile(latencies, 95) if latencies else 0
            p99_latency = np.percentile(latencies, 99) if latencies else 0
            throughput = successful / total_time if total_time > 0 else 0
            error_rate = failed / (successful + failed) * 100 if (successful + failed) > 0 else 0
            
            result = BenchmarkResult(
                name=f"{complexity.title()} Complexity",
                total_requests=requests_to_run,
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                avg_latency=avg_latency,
                min_latency=min_latency,
                max_latency=max_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                throughput=throughput,
                cpu_usage=(cpu_start + cpu_end) / 2,
                memory_usage=(memory_start + memory_end) / 2,
                error_rate=error_rate,
                timestamp=datetime.now().isoformat()
            )
            
            self.results.append(result)
            logger.info(f"{complexity} complexity benchmark completed: {throughput:.2f} RPS")
    
    async def benchmark_model_loading(self):
        """Benchmark model loading performance"""
        logger.info("Running model loading benchmark...")
        
        models_to_test = ["gpt2", "llama3.1:8b"]
        
        for model_name in models_to_test:
            logger.info(f"Testing {model_name} loading...")
            
            async with httpx.AsyncClient(timeout=self.config.timeout * 3) as client:
                # Unload model first (if loaded)
                await client.post(
                    f"{self.config.base_url}/models/unload",
                    params={"model_name": model_name},
                    headers={"Authorization": f"Bearer {self._get_test_token()}"}
                )
                
                # Measure loading time
                start_time = time.time()
                
                try:
                    response = await client.post(
                        f"{self.config.base_url}/models/load",
                        params={"model_name": model_name, "backend": "ollama"},
                        headers={"Authorization": f"Bearer {self._get_test_token()}"}
                    )
                    
                    load_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        logger.info(f"{model_name} loaded in {load_time:.2f}s")
                        
                        # Test inference performance
                        inference_times = []
                        for _ in range(5):
                            inference_start = time.time()
                            await self._make_reasoning_request(client, "Quick test", model_name)
                            inference_time = time.time() - inference_start
                            inference_times.append(inference_time)
                        
                        avg_inference_time = statistics.mean(inference_times)
                        logger.info(f"{model_name} avg inference time: {avg_inference_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
    
    async def _make_reasoning_request(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        model_backend: str = "ollama"
    ) -> bool:
        """Make a reasoning request and return success status"""
        try:
            response = await client.post(
                f"{self.config.base_url}/reason",
                json={
                    "prompt": prompt,
                    "max_depth": 3,
                    "num_chains": 4,
                    "temperature": 0.7,
                    "validate": False,  # Skip validation for speed
                    "model_backend": model_backend
                },
                headers={"Authorization": f"Bearer {self._get_test_token()}"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return False
    
    def _get_test_token(self) -> str:
        """Get test authentication token"""
        # For testing, create a simple token or use API key
        # In production, this would be obtained from login
        return "test_token_for_benchmarking"
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        if not self.results:
            logger.warning("No benchmark results to report")
            return
        
        # Create results directory
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate text report
        self._generate_text_report(results_dir / f"benchmark_report_{timestamp}.txt")
        
        # Generate JSON report
        self._generate_json_report(results_dir / f"benchmark_results_{timestamp}.json")
        
        # Generate plots
        self._generate_plots(results_dir / f"benchmark_plots_{timestamp}.png")
        
        logger.info(f"Benchmark report generated in {results_dir}")
    
    def _generate_text_report(self, filepath: Path):
        """Generate text report"""
        with open(filepath, 'w') as f:
            f.write("DPS Performance Benchmark Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Base URL: {self.config.base_url}\n")
            f.write(f"  - Total Requests: {self.config.num_requests}\n")
            f.write(f"  - Concurrent Users: {self.config.concurrent_users}\n")
            f.write(f"  - Timeout: {self.config.timeout}s\n\n")
            
            # Summary table
            headers = ["Test", "Total", "Success", "Failed", "Avg Latency", "P95", "Throughput", "Error %"]
            rows = []
            
            for result in self.results:
                rows.append([
                    result.name,
                    result.total_requests,
                    result.successful_requests,
                    result.failed_requests,
                    f"{result.avg_latency:.3f}s",
                    f"{result.p95_latency:.3f}s",
                    f"{result.throughput:.2f} RPS",
                    f"{result.error_rate:.1f}%"
                ])
            
            f.write(tabulate(rows, headers=headers, tablefmt="grid"))
            f.write("\n\n")
            
            # Detailed results
            for result in self.results:
                f.write(f"## {result.name}\n")
                f.write(f"- Total Time: {result.total_time:.2f}s\n")
                f.write(f"- Min Latency: {result.min_latency:.3f}s\n")
                f.write(f"- Max Latency: {result.max_latency:.3f}s\n")
                f.write(f"- P99 Latency: {result.p99_latency:.3f}s\n")
                f.write(f"- CPU Usage: {result.cpu_usage:.1f}%\n")
                f.write(f"- Memory Usage: {result.memory_usage:.1f}%\n")
                f.write("\n")
    
    def _generate_json_report(self, filepath: Path):
        """Generate JSON report"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "results": [asdict(result) for result in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_plots(self, filepath: Path):
        """Generate performance plots"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("DPS Performance Benchmark Results")
        
        # Throughput comparison
        names = [r.name for r in self.results]
        throughputs = [r.throughput for r in self.results]
        
        axes[0, 0].bar(names, throughputs)
        axes[0, 0].set_title("Throughput (RPS)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Latency comparison
        avg_latencies = [r.avg_latency for r in self.results]
        p95_latencies = [r.p95_latency for r in self.results]
        
        x = np.arange(len(names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, avg_latencies, width, label='Avg')
        axes[0, 1].bar(x + width/2, p95_latencies, width, label='P95')
        axes[0, 1].set_title("Latency Comparison")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(names, rotation=45)
        axes[0, 1].legend()
        
        # Error rates
        error_rates = [r.error_rate for r in self.results]
        axes[1, 0].bar(names, error_rates)
        axes[1, 0].set_title("Error Rates (%)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Resource usage
        cpu_usage = [r.cpu_usage for r in self.results]
        memory_usage = [r.memory_usage for r in self.results]
        
        x = np.arange(len(names))
        axes[1, 1].bar(x - width/2, cpu_usage, width, label='CPU %')
        axes[1, 1].bar(x + width/2, memory_usage, width, label='Memory %')
        axes[1, 1].set_title("Resource Usage")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(names, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()


async def main():
    """Run performance benchmarks"""
    config = BenchmarkConfig(
        base_url="http://localhost:8000",
        num_requests=50,  # Reduced for testing
        concurrent_users=5,
        timeout=30.0
    )
    
    benchmark = PerformanceBenchmark(config)
    results = await benchmark.run_all_benchmarks()
    
    print("\nBenchmark Summary:")
    print("-" * 50)
    for result in results:
        print(f"{result.name:<20} | {result.throughput:>8.2f} RPS | {result.avg_latency:>8.3f}s avg")


if __name__ == "__main__":
    asyncio.run(main())