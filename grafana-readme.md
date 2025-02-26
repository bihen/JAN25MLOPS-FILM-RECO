# Grafana Setup for Monitoring CPU and RAM Usage

## 1. Access Grafana

Grafana is already running as a service in your `docker-compose.yml` file. To access it:

- Open your browser and navigate to [**http://localhost:3001**](http://localhost:3001).
- Log in with the default credentials:
  - **Username:** `admin`
  - **Password:** `admin` (or the one set in your `docker-compose.yml`).

## 2. Add Prometheus as a Data Source

Prometheus is also running as a service and is scraping metrics from cAdvisor. To connect it to Grafana:

- In Grafana, go to **Configuration → Data Sources**.
- Click **Add data source** and select **Prometheus**.
- Set the **URL** to `http://prometheus:9090`.
- Click **Save & Test** to verify that Grafana can connect to Prometheus.

## 3. Create a Dashboard for CPU and RAM Monitoring

You have two options:

### Option 1: Import a Pre-Built Dashboard

- Go to **Grafana → Dashboards**.
- Click the **+ (Create) → Import** option.
- Use one of the official cAdvisor dashboards from the Grafana Dashboard Library:
  - **Docker Monitoring Dashboard** (ID: `893`)
  - **cAdvisor Dashboard** (ID: `13656`)
- Paste the ID and click **Load**.
- Select **Prometheus** as the data source and click **Import**.

### Option 2: Create a Custom Dashboard

1. Click **+ (Create) → Dashboard**.
2. Click **Add a new panel**.
3. **For CPU Usage**, enter this PromQL query:
   ```
   rate(container_cpu_usage_seconds_total{container_label_com_docker_compose_service="bentoml"}[1m])
   ```
   - This shows the per-second CPU usage rate for your BentoML container over the last minute.
4. **For Memory Usage**, enter this PromQL query:
   ```
   container_memory_usage_bytes{container_label_com_docker_compose_service="bentoml"}
   ```
   - This shows the real-time memory usage of your BentoML container.
5. Click **Apply** and repeat the process to add more panels if needed.
6. Save the dashboard.

## 4. Verify Data is Displayed

- Open your new dashboard.
- You should see live CPU and RAM usage statistics.
- If no data appears, ensure cAdvisor is running (`docker ps`) and check Prometheus (`http://localhost:9090/targets`).

## 5. Restarting the Services (If Needed)

If any service is not running, restart the stack:

```bash
docker-compose down && docker-compose up -d
```

## Notes

- cAdvisor is collecting metrics for all running containers.
- You can add more panels to track additional services.
- The dashboard updates every few seconds based on the scrape interval in `prometheus.yml`.



