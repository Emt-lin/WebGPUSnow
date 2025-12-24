<?php
/**
 * WebGPU Snow - 使用 WebGPU 实现的下雪粒子效果插件
 *
 * @package WebGPUSnow
 * @author wyh
 * @version 1.0.0
 * @link https://www.pslanys.com
 */

class WebGPUSnow_Plugin implements Typecho_Plugin_Interface
{
    /**
     * 激活插件
     *
     * @return void
     */
    public static function activate()
    {
        Typecho_Plugin::factory('Widget_Archive')->footer = __CLASS__ . '::renderSnow';
    }

    /**
     * 禁用插件
     *
     * @return void
     */
    public static function deactivate()
    {
        // 无需特殊清理
    }

    /**
     * 插件配置面板
     *
     * @param Typecho_Widget_Helper_Form $form 配置表单
     * @return void
     */
    public static function config(Typecho_Widget_Helper_Form $form)
    {
        // 粒子数量配置
        $particleCount = new Typecho_Widget_Helper_Form_Element_Text(
            'particleCount',
            null,
            '10000',
            _t('粒子数量'),
            _t('设置雪花粒子的数量，建议范围 5000-30000')
        );
        $form->addInput($particleCount);

        // 下落速度配置
        $snowSpeed = new Typecho_Widget_Helper_Form_Element_Text(
            'snowSpeed',
            null,
            '1.0',
            _t('下落速度'),
            _t('雪花下落速度系数，1.0 为正常速度')
        );
        $form->addInput($snowSpeed);

        // 风力配置
        $windForce = new Typecho_Widget_Helper_Form_Element_Text(
            'windForce',
            null,
            '0.02',
            _t('风力系数'),
            _t('水平风力大小，正值向右，负值向左，0 为无风')
        );
        $form->addInput($windForce);

        // 雪花大小范围
        $sizeMin = new Typecho_Widget_Helper_Form_Element_Text(
            'sizeMin',
            null,
            '3',
            _t('最小雪花大小 (像素)'),
            _t('雪花最小尺寸')
        );
        $form->addInput($sizeMin);

        $sizeMax = new Typecho_Widget_Helper_Form_Element_Text(
            'sizeMax',
            null,
            '12',
            _t('最大雪花大小 (像素)'),
            _t('雪花最大尺寸')
        );
        $form->addInput($sizeMax);

        // 画布层级
        $zIndex = new Typecho_Widget_Helper_Form_Element_Text(
            'zIndex',
            null,
            '9999',
            _t('画布层级 (z-index)'),
            _t('雪花层的 CSS z-index 值')
        );
        $form->addInput($zIndex);

        // 降级模式
        $useFallback = new Typecho_Widget_Helper_Form_Element_Radio(
            'useFallback',
            array(
                'auto' => _t('自动检测 (推荐)'),
                'webgpu' => _t('强制使用 WebGPU'),
                'canvas' => _t('强制使用 Canvas 2D')
            ),
            'auto',
            _t('渲染模式'),
            _t('选择雪花的渲染方式')
        );
        $form->addInput($useFallback);

        // 是否启用
        $enabled = new Typecho_Widget_Helper_Form_Element_Radio(
            'enabled',
            array(
                '1' => _t('启用'),
                '0' => _t('禁用')
            ),
            '1',
            _t('启用状态'),
            _t('是否启用下雪效果')
        );
        $form->addInput($enabled);
    }

    /**
     * 个人用户配置面板
     *
     * @param Typecho_Widget_Helper_Form $form 配置表单
     * @return void
     */
    public static function personalConfig(Typecho_Widget_Helper_Form $form)
    {
        // 不需要个人配置
    }

    /**
     * 渲染下雪效果
     *
     * @return void
     */
    public static function renderSnow()
    {
        $options = Helper::options()->plugin('WebGPUSnow');

        // 检查是否启用
        if ((string)$options->enabled !== '1') {
            return;
        }

        // Reason: 使用 trim + 空字符串判断，避免 0 等合法值被误判为"未设置"
        $particleCountRaw = trim((string)$options->particleCount);
        $particleCount = ($particleCountRaw === '') ? 10000 : max(1, intval($particleCountRaw));

        $snowSpeedRaw = trim((string)$options->snowSpeed);
        $snowSpeed = ($snowSpeedRaw === '') ? 1.0 : floatval($snowSpeedRaw);
        if ($snowSpeed <= 0) {
            $snowSpeed = 1.0;
        }

        $windForceRaw = trim((string)$options->windForce);
        $windForce = ($windForceRaw === '') ? 0.02 : floatval($windForceRaw);

        $sizeMinRaw = trim((string)$options->sizeMin);
        $sizeMaxRaw = trim((string)$options->sizeMax);
        $sizeMin = ($sizeMinRaw === '') ? 3 : max(0.1, floatval($sizeMinRaw));
        $sizeMax = ($sizeMaxRaw === '') ? 12 : max(0.1, floatval($sizeMaxRaw));
        if ($sizeMax < $sizeMin) {
            $tmp = $sizeMin;
            $sizeMin = $sizeMax;
            $sizeMax = $tmp;
        }

        $zIndexRaw = trim((string)$options->zIndex);
        $zIndex = ($zIndexRaw === '') ? 9999 : intval($zIndexRaw);

        $useFallback = (string)$options->useFallback;
        if (!in_array($useFallback, array('auto', 'webgpu', 'canvas'), true)) {
            $useFallback = 'auto';
        }

        // Reason: workerUrl 注入到 window.SNOW_CONFIG，供 snow.js 创建独立 Worker 使用
        $assetUrlRaw = rtrim(Helper::options()->pluginUrl, '/\\') . '/WebGPUSnow/assets';

        $config = array(
            'particleCount' => $particleCount,
            'snowSpeed' => $snowSpeed,
            'windForce' => $windForce,
            'sizeMin' => $sizeMin,
            'sizeMax' => $sizeMax,
            'zIndex' => $zIndex,
            'useFallback' => $useFallback,
            'workerUrl' => $assetUrlRaw . '/snow.worker.js'
        );

        // 获取插件 URL 并转义（仅用于 HTML 标签）
        $assetUrlEsc = htmlspecialchars($assetUrlRaw, ENT_QUOTES, 'UTF-8');

        // 输出配置和脚本
        echo '<script>window.SNOW_CONFIG = ' . json_encode($config) . ';</script>' . "\n";
        echo '<link rel="stylesheet" href="' . $assetUrlEsc . '/snow.css">' . "\n";
        echo '<script type="module" src="' . $assetUrlEsc . '/snow.js"></script>' . "\n";
    }
}
