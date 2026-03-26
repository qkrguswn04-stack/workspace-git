$(document).ready(function() {

    // var newImageUrl = "/static/" + getRandomInt(1, 46) + ".png";

    // .cart-page-outside 클래스의 background 속성 업데이트
    $(".cart-page-outside").css("background", `#c72320 url("../static/img/1.webp") no-repeat center`);
    $(".cart-page-outside").css("background-size", "cover");
    $(".cart-page-outside").css("width", "100%");
    $(".cart-page-outside").css("height", "100%");


    var isCardOpened = false; // 첫 번째 클릭 여부를 확인하는 플래그 변수
    var $clickMe = $('.click-icon'),
        $card = $('.card');
    
    $card.on('click', function() {
            $(this).toggleClass('is-opened');
            $clickMe.toggleClass('is-hidden');
            if (!isCardOpened) {
                var msg1 = $("#letter").attr('data-message1').split(',').map(s => s.trim().replace(/^"|"$/g, ''));
                var msg2 = $("#letter").attr('data-message2').split(',').map(s => s.trim().replace(/^"|"$/g, ''));
                var msg3 = $("#letter").attr('data-message3').split(',').map(s => s.trim().replace(/^"|"$/g, ''));
                isCardOpened = true; // 첫 번째 클릭 이후 true로 설정
                makecard(msg1, msg2, msg3);
            }
      });
  });
  function getRandomInt(min, max) {
      return Math.floor(Math.random() * (max - min + 1)) + min;
  }
  function makecard(msg1, msg2, msg3){
    var options = {
      strings: msg1,
      typeSpeed: 80, // 한 글자당 타이핑 속도 (밀리초)
      backSpeed: 30, // 한 글자당 삭제 속도 (밀리초)
      loop: false, // 반복 여부
      onComplete: function(self){ self.cursor.remove();
                        new Typed('#typing-text2', option1);
      }
  };
  var option1 = {
    strings: msg2,
    typeSpeed: 80, // 한 글자당 타이핑 속도 (밀리초)
    backSpeed: 50, // 한 글자당 삭제 속도 (밀리초)
    loop: false,
    onComplete : function(self){
        self.cursor.remove();
        // 마지막 문장 뒤에 마침표 추가하고, 더 이상 타이핑하지 않음
        onCompleteLastTyping();
    }
  }
  var typed = new Typed('#typing-text1', options);
  function onCompleteLastTyping() {
        // 다른 스타일 또는 옵션으로 새 Typed 인스턴스 생성
        new Typed("#typing-text3", {
            strings: msg3,
            typeSpeed: 90,
            backSpeed: 30,
            showCursor: true,
            cursorChar: "_",
            onComplete : function(self){
                $('#restartButton').show();
                self.cursor.remove();
            }
        });
    }
  }
