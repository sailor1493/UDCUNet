# train 중 validation

+ `basicsr/train.py:192`: `model.validation(...)`
  + `basicsr/models/base_model.py:36-48`: `self.dist_validation()` 혹은 `self.nondist_validation`을 호출함
    + `basicsr/models/udc_model.py:232`: `def dist_validation()` -> `self.nondist_validation()`을 호출함
    + `basicsr/models/udc_model.py:236`: `def nondist_validation()` -> 별도 섹션에서 설명

# test

+ `basicsr/test.py:40`: `model.validation(...)`
  + 이하 과정은 위와 같음

# `UDCModel.nondist_validation`

Progressbar, 이미지 저장 관련된 부분은 코드에서 `...` 처리했습니다.

```Python
def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
    ... 초기화

    metric_data = dict()

    ... 상태 바

    for idx, val_data in enumerate(dataloader):
        # Forward Pass를 돌리는 부분
        img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        self.feed_data(val_data)
        self.test()

        visuals = self.get_current_visuals()
        # sr_npy_0 = np.array((visuals['result']))
        # sr_npy = sr_npy_0.copy()
        # sr_npy = np.clip(sr_npy, 0, 500) 아래에서 참조하지 않는 변수이기 때문에 주석처리해도 무방함

        #
        # 우리 팀의 주요 관심사 부분
        # 

        sr_img_metric = visuals['result'].clone()
        sr_img_metric = torch.clamp(sr_img_metric, 0, 500)
        sr_img_metric = tone_map(sr_img_metric)
        sr_img_metric = tensor2img([sr_img_metric])
        sr_img = tensor2img([tone_map(torch.clamp(visuals['result'], 0, 500))])
        metric_data['img'] = sr_img_metric
        if 'gt' in visuals:
            gt_img_metric = tone_map(visuals['gt']).clone()
            gt_img_metric = tensor2img([gt_img_metric])
            # gt_img = tensor2img([visuals['gt']])
            metric_data['img2'] = gt_img_metric
            del self.gt

        # tentative for out of GPU memory 로직과 관련없음
        ...

        if save_img:
            ... 사진을 저장한다

        if with_metrics:
            # calculate metrics
            for name, opt_ in self.opt['val']['metrics'].items():
                self.metric_results[name] += calculate_metric(metric_data, opt_)
        if use_pbar:
            ... 상태바 업데이트
    if use_pbar:
        pbar.close()

    if with_metrics:
        for metric in self.metric_results.keys():
            self.metric_results[metric] /= (idx + 1)
            # update the best metric result
            self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
```

# Optimize

+ 하는 일 : train 시에는 feed data -> optimize_param -> iter 주기가 맞으면 validation을 돌림
+ 

```Python
def optimize_parameters(self, current_iter):
    self.optimizer_g.zero_grad()
    self.output = self.net_g(self.lq)  # here we run forward pass

    l_total = 0
    loss_dict = OrderedDict()
    # pixel loss
    if self.cri_pix:
        l_pix = self.cri_pix(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix
    # perceptual loss
    if self.cri_perceptual:
        l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        if l_percep is not None:
            l_total += l_percep
            loss_dict['l_percep'] = l_percep
        if l_style is not None:
            l_total += l_style
            loss_dict['l_style'] = l_style

    l_total.backward()
    self.optimizer_g.step()

    self.log_dict = self.reduce_loss_dict(loss_dict)

    if self.ema_decay > 0:
        self.model_ema(decay=self.ema_decay)
```
