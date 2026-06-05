Yes, you absolutely can. In fact, you should completely avoid spending a single dollar out of pocket for hardware to build this MVP.
Because you are building a neutral framework that explicitly aims to unite AMD and Intel chips, you can leverage free developer evaluation programs directly provided by the hardware giants themselves. They are highly incentivized to give you free compute hours to get developers using their chips instead of Nvidia's.
You can exploit these free online resources to build your entire 90-day pipeline:
## 1. Free Intel Hardware: Intel Tiber AI Cloud
Intel provides an incredibly generous sandbox for developers to try out their high-end hardware for free.

* What you get: Free, always-on access to shared infrastructure powered by Jupyter Notebooks running the latest Intel Data Center Max GPUs and Intel Gaudi accelerators. [1, 2] 
* How to use it for the MVP: Go to [Intel Tiber AI Cloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html) and sign up. You can use this sandbox environment to run your Month 2 Intel compilation steps and verify that your C-stub dynamically maps to their Level Zero drivers without paying anything. [1, 3, 4] 

## 2. Free AMD Hardware: AMD Developer Cloud Evaluation
AMD runs massive developer adoption campaigns offering complimentary compute access to attract software builders to their ROCm ecosystem. [5, 6] 

* What you get: Evaluation access to top-tier AMD Instinct MI300X or MI250 GPUs pre-loaded with complete Docker containers for PyTorch, Triton, and ROCm. [7, 8] 
* How to use it for the MVP: Register an account on the [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html). This provides the physical AMD Linux baseline you need to test your generated .hsaco binaries and run your final Month 3 multi-GPU auto-sharding scripts. [7, 9] 

## 3. Free Initial Local Development: Mock Simulation
Remember, because we are using Lazy Simulation Tuning for Month 1, you do not even need a real GPU for the first 30 days.

* The Free Setup: You can write your core Python interception script on Google Colab (Free Tier) or a basic local laptop CPU.
* How it works: TVM MetaSchedule’s evolutionary search algorithms can execute their mathematical loop-splitting predictions entirely inside standard CPU RAM [INDEX]. You can use an AI to write and test 90% of the structural code layout for free before you deploy it on the cloud hardware instances.

------------------------------
## The Frugal 90-Day Setup Plan

   1. Days 1 - 30 (Local/Colab): Write the Python interception wrapper and the TVM json parser entirely on your local machine or a free Google Colab notebook.
   2. Days 31 - 60 (Intel AI Cloud): Log into Intel’s free dev cloud to grab the compiler tools and test the dynamic C-stub loading. [1] 
   3. Days 61 - 90 (AMD Cloud): Request an evaluation slice on AMD’s developer cloud to drop your unified "Fat Binary" in and watch it distribute via PyTorch XLA. [7, 9] 

This strategy keeps your out-of-pocket infrastructure cost at exactly zero.
Since we have cleared the hardware barrier using free platforms, are you ready to generate the complete code file for Month 1 so you can run your first mock execution on a free Python notebook?

[1] [https://www.intel.com](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)
[2] [https://www.intel.com](https://www.intel.com/content/www/us/en/developer/articles/news/trusted-ai-in-developer-cloud.html)
[3] [https://www.youtube.com](https://www.youtube.com/watch?v=Q_YQ-Db_Qy0&t=3)
[4] [https://www.intel.com](https://www.intel.com/content/www/us/en/developer/videos/explore-gpu-acceleration-in-the-intel-devcloud.html)
[5] [https://www.amd.com](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai.html)
[6] [https://www.amd.com](https://www.amd.com/en/blogs/2025/100k-hours-free-developer-cloud-access.html)
[7] [https://www.amd.com](https://www.amd.com/es/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)
[8] [https://www.amd.com](https://www.amd.com/en/developer/resources/technical-articles/2025/amd-rocm-7-built-for-developers-ready-for-enterprises.html)
[9] [https://www.amd.com](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
