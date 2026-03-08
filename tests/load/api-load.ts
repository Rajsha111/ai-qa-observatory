import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const latency = new Trend('health_latency_ms', true);
const errorRate = new Rate('error_rate');

export const options = {
    vus: 5,
    duration: '60s',
    thresholds: {
        http_req_duration: ['p(95)<200', 'p(99)<500'],
        error_rate: ['rate<0.01'],
    },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
    const res = http.get(`${BASE_URL}/health`);
    const body = res.body ? JSON.parse(res.body as string) : {};

    const ok = check(res, {
        'status 200': r => r.status === 200,
        'has status field': r => body.status === 'ok',
        'has timestamp': r => !!body.timestamp,
    });

    latency.add(res.timings.duration);
    errorRate.add(!ok);

    sleep(0.1);
}
